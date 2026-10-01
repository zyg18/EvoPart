// Multilevel recursive-bisection hypergraph partitioner (cut-net metric).
//
// Usage: ./our_hparter <input.hgr> --k=K --eps=EPS --seed=S --out=<partition file>
//
// Per bisection: heavy-edge first-choice clustering (parallel nets merged on
// contraction) -> greedy hypergraph growing + FM on the coarsest level ->
// boundary FM at every level while uncoarsening, then V-cycles.  k-way comes
// from recursive bisection; a net that is not wholly inside a block is dropped
// from that block's sub-problem, which is exact for the cut-net metric.  The
// final blocks are then re-partitioned pair by pair, repeatedly while it pays,
// which is the only way to revisit a split the bisection tree committed to
// before its children existed.  Every finest-level bisection is additionally
// polished by max-flow min-cut refinement on the region around the cut, a
// neighbourhood FM cannot reach.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace std;
typedef vector<int> VI;
typedef long long ll;

static const int FMLIM = 10000;  // nets above this size: pin counts kept exact, gains not propagated
static const int RATELIM = 400;  // nets above this size are ignored when rating clustering partners
static const int RATEWORK = 600; // pin budget per vertex when rating partners
static const int STAGE[3] = {16, 128, RATELIM};   // net-size bands, smallest rated first
static const int COARSEST = 200;
static const int MAXLV = 64;     // levels are reserve()d, so level vectors never reallocate
static const int UPASS = 6;      // FM passes per uncoarsening level
static const int RMAX = 24;      // hard cap on multilevel restarts per bisection
static const int MOVELIM = 4096; // max pins moved by one compound net move
static const int HECMIN = 8;     // net-size window worth contracting whole (see hecPhase)
static const int HECMAX = 512;
static const int FLOWALPHA = 16;  // flow region: alpha x balance slack, halved until the min cut fits
static const int FLOWCAP = 20000; // region vertices per side, bounds one max-flow
static const int FLOWEXP = 2000;  // nets above this size do not grow the region
static const int INFCAP = 1 << 29;
static const int SWEEPS = 3;      // deep pair sweeps allowed inside one budget window

static double nowSec() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}
static double g_deadline = 1e18;
static inline bool outOfTime() { return nowSec() > g_deadline; }
static double g_tCoarsen = 0, g_tInit = 0, g_tRefine = 0, g_tPair = 0, g_tFlow = 0;
static int g_levels = 0, g_bisections = 0, g_restarts = 0, g_vcyc = 0, g_pairGain = 0, g_runs = 0;
static ll g_netGain = 0, g_netMoves = 0, g_flowGain = 0;
static int g_deepPairs = 0, g_scratchWins = 0, g_flowRuns = 0;
static bool g_useHEC = false;    // set where pins concentrate in large nets
static int g_hecWins = 0, g_hecTries = 0;

static inline uint64_t xr(uint64_t &s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

struct HGraph {
    int n = 0, m = 0;
    VI eptr, eind;   // net -> pins
    VI vptr, vind;   // vertex -> nets
    VI nw, vw;
    ll totW = 0;
    void buildInc() {
        vptr.assign(n + 1, 0);
        for (size_t i = 0; i < eind.size(); ++i) ++vptr[eind[i] + 1];
        for (int i = 0; i < n; ++i) vptr[i + 1] += vptr[i];
        vind.resize(eind.size());
        VI pos(vptr.begin(), vptr.end() - 1);
        for (int e = 0; e < m; ++e)
            for (int j = eptr[e]; j < eptr[e + 1]; ++j) vind[pos[eind[j]]++] = e;
        totW = 0;
        for (int v = 0; v < n; ++v) totW += vw[v];
    }
};

// Indexed binary max-heap over an external key array; ties broken by smaller id.
struct Heap {
    VI h, pos;
    const VI *key = nullptr;
    int sz = 0;
    void init(int n, const VI *k) { key = k; pos.assign(n, -1); h.clear(); sz = 0; }
    void clear() { for (int i = 0; i < sz; ++i) pos[h[i]] = -1; h.clear(); sz = 0; }
    inline bool better(int a, int b) const {
        int ka = (*key)[a], kb = (*key)[b];
        return ka > kb || (ka == kb && a < b);
    }
    void up(int i) {
        while (i) {
            int p = (i - 1) >> 1;
            if (!better(h[i], h[p])) break;
            swap(h[i], h[p]); pos[h[i]] = i; pos[h[p]] = p; i = p;
        }
    }
    void down(int i) {
        for (;;) {
            int l = 2 * i + 1, r = l + 1, b = i;
            if (l < sz && better(h[l], h[b])) b = l;
            if (r < sz && better(h[r], h[b])) b = r;
            if (b == i) break;
            swap(h[i], h[b]); pos[h[i]] = i; pos[h[b]] = b; i = b;
        }
    }
    void push(int v) {
        if (pos[v] >= 0) { up(pos[v]); down(pos[v]); return; }
        h.push_back(v); pos[v] = sz++; up(sz - 1);
    }
    int top() const { return sz ? h[0] : -1; }
    void pop() {
        if (!sz) return;
        pos[h[0]] = -1;
        int last = h[sz - 1]; h.pop_back(); --sz;
        if (sz) { h[0] = last; pos[last] = 0; down(0); }
    }
};

static bool readHgr(const char *path, HGraph &g)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) { fclose(f); return false; }
    vector<char> buf((size_t)len + 1);
    size_t got = fread(buf.data(), 1, (size_t)len, f);
    fclose(f);
    buf[got] = '\n';
    char *p = buf.data(), *end = buf.data() + got + 1;
    int m = 0, n = 0;
    while (p < end) {
        char *le = (char *)memchr(p, '\n', end - p);
        if (!le) le = end;
        char *q = p;
        p = (le < end) ? le + 1 : end;
        while (q < le && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
        if (q >= le || *q == '%') continue;
        m = 0; while (q < le && *q >= '0' && *q <= '9') m = m * 10 + (*q++ - '0');
        while (q < le && (*q == ' ' || *q == '\t')) ++q;
        n = 0; while (q < le && *q >= '0' && *q <= '9') n = n * 10 + (*q++ - '0');
        break;
    }
    if (m <= 0 || n <= 0) return false;
    g.n = n;
    g.eptr.clear(); g.eptr.reserve(m + 1); g.eptr.push_back(0);
    g.eind.clear(); g.eind.reserve((size_t)m * 4);
    int read = 0;
    while (p < end && read < m) {
        char *le = (char *)memchr(p, '\n', end - p);
        if (!le) le = end;
        char *q = p;
        p = (le < end) ? le + 1 : end;
        while (q < le && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
        if (q >= le || *q == '%') continue;
        while (q < le) {
            while (q < le && (*q < '0' || *q > '9')) ++q;
            if (q >= le) break;
            int v = 0;
            while (q < le && *q >= '0' && *q <= '9') v = v * 10 + (*q++ - '0');
            if (v >= 1 && v <= n) g.eind.push_back(v - 1);
        }
        g.eptr.push_back((int)g.eind.size());
        ++read;
    }
    g.m = (int)g.eptr.size() - 1;
    g.nw.assign(g.m, 1);
    g.vw.assign(n, 1);
    g.buildInc();
    return true;
}

// Whole-net contraction.  The pairwise rating weights a net by 1/(|e|-1), so a
// 200-pin net never decides a match, and FM sees zero gain on a net with many
// pins on both sides -- yet an uncut net of any size is worth one unit.  Merging
// all pins of such a net into one cluster is the only way the multilevel scheme
// can keep it whole.  Largest feasible net first, all of its pins still free.
static void hecPhase(const HGraph &g, VI &cmap, vector<ll> &cw, int &nc,
                     ll maxNodeW, const signed char *restr)
{
    VI cand;
    for (int e = 0; e < g.m; ++e) {
        int sz = g.eptr[e + 1] - g.eptr[e];
        if (sz >= HECMIN && sz <= HECMAX && (ll)sz <= maxNodeW) cand.push_back(e);
    }
    if (cand.empty()) return;
    sort(cand.begin(), cand.end(), [&](int a, int b) {
        int sa = g.eptr[a + 1] - g.eptr[a], sb = g.eptr[b + 1] - g.eptr[b];
        if (sa != sb) return sa > sb;
        return a < b;
    });
    int budget = g.n / 2;   // the pairwise phase must keep enough room to work in
    for (size_t i = 0; i < cand.size() && budget >= HECMIN; ++i) {
        int e = cand[i], b = g.eptr[e], en = g.eptr[e + 1];
        if (en - b > budget) continue;
        signed char p0 = restr ? restr[g.eind[b]] : (signed char)0;
        ll w = 0;
        bool ok = true;
        for (int j = b; j < en; ++j) {
            int u = g.eind[j];
            if (cmap[u] >= 0 || (restr && restr[u] != p0)) { ok = false; break; }
            w += g.vw[u];
            if (w > maxNodeW) { ok = false; break; }
        }
        if (!ok) continue;
        int c = nc++;
        cw.push_back(w);
        for (int j = b; j < en; ++j) cmap[g.eind[j]] = c;
        budget -= en - b;
    }
}

// First-choice clustering: each unassigned vertex joins the best-rated partner
// cluster whose weight still fits maxNodeW.  With restr != null only vertices
// of the same block may be merged (V-cycle clustering).
static int clusterLevel(const HGraph &g, VI &cmap, ll maxNodeW, uint64_t &rs,
                        const signed char *restr, bool hec)
{
    const int n = g.n;
    cmap.assign(n, -1);
    VI order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    for (int i = n - 1; i > 0; --i) { int j = (int)(xr(rs) % (uint64_t)(i + 1)); swap(order[i], order[j]); }
    vector<float> score(n, 0.f);
    VI touched;
    touched.reserve(256);
    vector<ll> cw;
    int nc = 0;
    if (hec) hecPhase(g, cmap, cw, nc, maxNodeW, restr);
    for (int idx = 0; idx < n; ++idx) {
        int v = order[idx];
        if (cmap[v] >= 0) continue;
        signed char pv = restr ? restr[v] : (signed char)0;
        int emin = -1, eminSz = INT_MAX;
        for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
            int e = g.vind[j];
            int sz = g.eptr[e + 1] - g.eptr[e];
            if (sz >= 2 && sz < eminSz) { eminSz = sz; emin = e; }
        }
        // Rate nets in ascending size bands: a 2-pin net identifies a partner,
        // a 200-pin net barely does, so the pin budget must reach the small
        // nets first.  The widest band runs only if nothing was rated at all.
        int work = 0;
        for (int st = 0; st < 3 && work < RATEWORK; ++st) {
            if (st == 2 && !touched.empty()) break;
            int lo = st ? STAGE[st - 1] : 1, hi = STAGE[st];
            for (int j = g.vptr[v]; j < g.vptr[v + 1] && work < RATEWORK; ++j) {
                int e = g.vind[j];
                int sz = g.eptr[e + 1] - g.eptr[e];
                if (sz <= lo || sz > hi) continue;
                work += sz;
                float w = (float)g.nw[e] / (float)(sz - 1);
                for (int t = g.eptr[e]; t < g.eptr[e + 1]; ++t) {
                    int u = g.eind[t];
                    if (u == v || (restr && restr[u] != pv)) continue;
                    if (score[u] == 0.f) touched.push_back(u);
                    score[u] += w;
                }
            }
        }
        int best = -1;
        float bestVal = 0.f;
        for (size_t i = 0; i < touched.size(); ++i) {
            int u = touched[i];
            ll wu = (cmap[u] >= 0) ? cw[cmap[u]] : (ll)g.vw[u];
            if ((ll)g.vw[v] + wu > maxNodeW) continue;
            float val = score[u] / (float)wu;   // multiplicative heavy-node penalty
            if (val > bestVal || (val == bestVal && best >= 0 && u < best)) { bestVal = val; best = u; }
        }
        for (size_t i = 0; i < touched.size(); ++i) score[touched[i]] = 0.f;
        touched.clear();
        if (best < 0 && emin >= 0) {
            // Only huge nets touch v: take the lightest partner in a seeded
            // window of the smallest of them, so clusters stay fine-grained.
            int b = g.eptr[emin], len = g.eptr[emin + 1] - b;
            int lim = min(len, 64), st = (int)(xr(rs) % (uint64_t)len);
            ll bw = LLONG_MAX;
            for (int q = 0; q < lim; ++q) {
                int x = b + st + q;
                if (x >= b + len) x -= len;
                int u = g.eind[x];
                if (u == v || (restr && restr[u] != pv)) continue;
                ll wu = (cmap[u] >= 0) ? cw[cmap[u]] : (ll)g.vw[u];
                if ((ll)g.vw[v] + wu <= maxNodeW && wu < bw) { bw = wu; best = u; }
            }
        }
        if (best >= 0) {
            int c;
            if (cmap[best] >= 0) { c = cmap[best]; cw[c] += g.vw[v]; }
            else { c = nc++; cw.push_back((ll)g.vw[best] + g.vw[v]); cmap[best] = c; }
            cmap[v] = c;
        }
    }
    for (int v = 0; v < n; ++v) if (cmap[v] < 0) { cmap[v] = nc++; cw.push_back(g.vw[v]); }
    return nc;
}

static void contract(const HGraph &g, const VI &cmap, int nc, HGraph &cg)
{
    cg.n = nc;
    cg.vw.assign(nc, 0);
    for (int v = 0; v < g.n; ++v) cg.vw[cmap[v]] += g.vw[v];
    VI stamp(nc, -1), tmp, ep, ei, ew;
    vector<uint64_t> hs;
    ep.push_back(0);
    ei.reserve(g.eind.size());
    for (int e = 0; e < g.m; ++e) {
        tmp.clear();
        for (int j = g.eptr[e]; j < g.eptr[e + 1]; ++j) {
            int c = cmap[g.eind[j]];
            if (stamp[c] != e) { stamp[c] = e; tmp.push_back(c); }
        }
        if (tmp.size() < 2) continue;
        sort(tmp.begin(), tmp.end());
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < tmp.size(); ++i) h = (h ^ (uint64_t)(tmp[i] + 1)) * 1099511628211ULL;
        hs.push_back(h);
        ew.push_back(g.nw[e]);
        for (size_t i = 0; i < tmp.size(); ++i) ei.push_back(tmp[i]);
        ep.push_back((int)ei.size());
    }
    int mm = (int)ew.size();
    VI idx(mm);
    for (int i = 0; i < mm; ++i) idx[i] = i;
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (hs[a] != hs[b]) return hs[a] < hs[b];
        int sa = ep[a + 1] - ep[a], sb = ep[b + 1] - ep[b];
        if (sa != sb) return sa < sb;
        int c = memcmp(&ei[ep[a]], &ei[ep[b]], (size_t)sa * sizeof(int));
        if (c != 0) return c < 0;
        return a < b;
    });
    vector<char> dead(mm, 0);
    for (int i = 0; i < mm;) {
        int a = idx[i], j = i + 1, sa = ep[a + 1] - ep[a];
        while (j < mm) {
            int b = idx[j];
            if (hs[b] != hs[a] || ep[b + 1] - ep[b] != sa) break;
            if (memcmp(&ei[ep[a]], &ei[ep[b]], (size_t)sa * sizeof(int)) != 0) break;
            ew[a] += ew[b];
            dead[b] = 1;
            ++j;
        }
        i = j;
    }
    cg.eptr.clear(); cg.eind.clear(); cg.nw.clear();
    cg.eptr.push_back(0);
    cg.eind.reserve(ei.size());
    for (int e = 0; e < mm; ++e) {
        if (dead[e]) continue;
        for (int j = ep[e]; j < ep[e + 1]; ++j) cg.eind.push_back(ei[j]);
        cg.eptr.push_back((int)cg.eind.size());
        cg.nw.push_back(ew[e]);
    }
    cg.m = (int)cg.nw.size();
}

// Two-way FM.  The cut delta is derived from pin counts (never from the gain
// array), so roll-back stays exact even for nets whose gains we do not update.
// Never returns a worse partition than it was given.
static ll fmRefine(const HGraph &g, vector<signed char> &part, ll maxW0, ll maxW1, int maxPasses)
{
    const int n = g.n;
    if (n == 0) return 0;
    VI pc0(g.m, 0), pc1(g.m, 0), gain(n, 0);
    vector<char> locked(n, 0);
    Heap h0, h1;
    h0.init(n, &gain); h1.init(n, &gain);
    ll w0 = 0, w1 = 0;
    for (int v = 0; v < n; ++v) { if (part[v]) w1 += g.vw[v]; else w0 += g.vw[v]; }
    VI moveLog;
    moveLog.reserve(1024);
    ll cut = 0;
    int stopLimit = (int)max(300LL, min((ll)n / 8, 10000LL));
    for (int pass = 0; pass < maxPasses; ++pass) {
        cut = 0;
        for (int e = 0; e < g.m; ++e) {
            int c0 = 0, c1 = 0;
            for (int j = g.eptr[e]; j < g.eptr[e + 1]; ++j) { if (part[g.eind[j]]) ++c1; else ++c0; }
            pc0[e] = c0; pc1[e] = c1;
            if (c0 && c1) cut += g.nw[e];
        }
        fill(locked.begin(), locked.end(), 0);
        h0.clear(); h1.clear();
        for (int v = 0; v < n; ++v) {
            int F = part[v], gg = 0;
            bool bnd = false;
            for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
                int e = g.vind[j];
                int cf = F ? pc1[e] : pc0[e], ct = F ? pc0[e] : pc1[e];
                if (cf == 1) gg += g.nw[e];
                if (ct == 0) gg -= g.nw[e]; else bnd = true;
            }
            gain[v] = gg;
            if (bnd) { if (F) h1.push(v); else h0.push(v); }
        }
        ll startCut = cut, bestCut = cut;
        size_t bestPrefix = 0;
        moveLog.clear();
        int sinceBest = 0, counter = 0;
        while (true) {
            if (((++counter) & 1023) == 0 && outOfTime()) break;
            int v0 = h0.top(), v1 = h1.top();
            bool ok0 = v0 >= 0 && (w1 + g.vw[v0] <= maxW1);
            bool ok1 = v1 >= 0 && (w0 + g.vw[v1] <= maxW0);
            int from;
            if (ok0 && ok1) {
                int a = gain[v0], b = gain[v1];
                if (a > b) from = 0;
                else if (b > a) from = 1;
                else from = (w0 * maxW1 >= w1 * maxW0) ? 0 : 1;
            } else if (ok0) from = 0;
            else if (ok1) from = 1;
            else break;
            int v = from ? v1 : v0;
            if (from) h1.pop(); else h0.pop();
            locked[v] = 1;
            int T = 1 - from;
            ll delta = 0;
            for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
                int e = g.vind[j];
                int f = from ? pc1[e] : pc0[e];
                int t = from ? pc0[e] : pc1[e];
                int w = g.nw[e], b = g.eptr[e], en = g.eptr[e + 1];
                bool wasCut = (f > 0 && t > 0);
                bool upd = (en - b) <= FMLIM;
                if (upd) {
                    if (t == 0) {
                        for (int q = b; q < en; ++q) {
                            int u = g.eind[q];
                            if (u != v && !locked[u]) { gain[u] += w; (part[u] ? h1 : h0).push(u); }
                        }
                    } else if (t == 1) {
                        for (int q = b; q < en; ++q) {
                            int u = g.eind[q];
                            if (u != v && part[u] == T) {
                                if (!locked[u]) { gain[u] -= w; (part[u] ? h1 : h0).push(u); }
                                break;
                            }
                        }
                    }
                }
                --f; ++t;
                if (upd) {
                    if (f == 0) {
                        for (int q = b; q < en; ++q) {
                            int u = g.eind[q];
                            if (u != v && !locked[u]) { gain[u] -= w; (part[u] ? h1 : h0).push(u); }
                        }
                    } else if (f == 1) {
                        for (int q = b; q < en; ++q) {
                            int u = g.eind[q];
                            if (u != v && part[u] == from) {
                                if (!locked[u]) { gain[u] += w; (part[u] ? h1 : h0).push(u); }
                                break;
                            }
                        }
                    }
                }
                if (from) { pc1[e] = f; pc0[e] = t; } else { pc0[e] = f; pc1[e] = t; }
                delta += (wasCut ? w : 0) - ((f > 0 && t > 0) ? w : 0);
            }
            part[v] = (signed char)T;
            if (from == 0) { w0 -= g.vw[v]; w1 += g.vw[v]; } else { w1 -= g.vw[v]; w0 += g.vw[v]; }
            cut -= delta;
            moveLog.push_back(v);
            if (cut < bestCut) { bestCut = cut; bestPrefix = moveLog.size(); sinceBest = 0; }
            else if (++sinceBest > stopLimit) break;
        }
        for (size_t i = moveLog.size(); i > bestPrefix; --i) {
            int v = moveLog[i - 1], F = part[v];
            part[v] = (signed char)(1 - F);
            if (F) { w1 -= g.vw[v]; w0 += g.vw[v]; } else { w0 -= g.vw[v]; w1 += g.vw[v]; }
        }
        cut = bestCut;
        if (bestCut >= startCut || outOfTime()) break;
    }
    return cut;
}

// Whole-net compound moves.  A net with many pins on both sides is a plateau
// for single-vertex FM: no one move changes the cut, only moving an entire side
// of the net does.  cnt[f] counts how many pin occurrences of f are about to
// move, so the exact delta over the touched nets costs O(pins(e) * degree).
// Only strictly improving, balance-feasible moves are applied.
static ll netMoveRefine(const HGraph &g, vector<signed char> &part, ll maxW0, ll maxW1, int rounds)
{
    const int n = g.n, m = g.m;
    if (n == 0 || m == 0) return 0;
    VI c0(m, 0), c1(m, 0), cnt(m, 0), vs(n, 0), S, touched;
    ll w0 = 0, w1 = 0;
    for (int v = 0; v < n; ++v) { if (part[v]) w1 += g.vw[v]; else w0 += g.vw[v]; }
    for (int e = 0; e < m; ++e) {
        int a = 0, b = 0;
        for (int j = g.eptr[e]; j < g.eptr[e + 1]; ++j) { if (part[g.eind[j]]) ++b; else ++a; }
        c0[e] = a; c1[e] = b;
    }
    ll gained = 0, work = 0, budget = 4 * (ll)g.eind.size() + 200000;
    int mark = 0;
    for (int r = 0; r < rounds; ++r) {
        ll round = 0;
        for (int e = 0; e < m; ++e) {
            if (c0[e] == 0 || c1[e] == 0) continue;
            if (work > budget) return gained;
            if ((e & 255) == 0 && outOfTime()) return gained;
            for (int side = 0; side < 2; ++side) {
                int cs = side ? c1[e] : c0[e];
                if (cs == 0 || cs > MOVELIM) continue;
                ++mark;
                S.clear();
                ll wS = 0;
                work += g.eptr[e + 1] - g.eptr[e];
                for (int j = g.eptr[e]; j < g.eptr[e + 1]; ++j) {
                    int v = g.eind[j];
                    if (part[v] != side || vs[v] == mark) continue;
                    vs[v] = mark; S.push_back(v); wS += g.vw[v];
                }
                if (S.empty()) continue;
                if (side == 0 ? (w1 + wS > maxW1) : (w0 + wS > maxW0)) continue;
                touched.clear();
                for (size_t i = 0; i < S.size(); ++i) {
                    int v = S[i];
                    work += g.vptr[v + 1] - g.vptr[v];
                    for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
                        int f = g.vind[j];
                        if (cnt[f]++ == 0) touched.push_back(f);
                    }
                }
                ll delta = 0;
                for (size_t i = 0; i < touched.size(); ++i) {
                    int f = touched[i], s = cnt[f];
                    int a0 = c0[f], a1 = c1[f];
                    int b0 = side ? a0 + s : a0 - s, b1 = side ? a1 - s : a1 + s;
                    bool was = (a0 > 0 && a1 > 0), now = (b0 > 0 && b1 > 0);
                    if (was && !now) delta += g.nw[f];
                    else if (!was && now) delta -= g.nw[f];
                }
                if (delta > 0) {
                    for (size_t i = 0; i < touched.size(); ++i) {
                        int f = touched[i], s = cnt[f];
                        if (side) { c1[f] -= s; c0[f] += s; } else { c0[f] -= s; c1[f] += s; }
                    }
                    for (size_t i = 0; i < S.size(); ++i) part[S[i]] = (signed char)(1 - side);
                    if (side) { w1 -= wS; w0 += wS; } else { w0 -= wS; w1 += wS; }
                    gained += delta; round += delta; ++g_netMoves;
                }
                for (size_t i = 0; i < touched.size(); ++i) cnt[touched[i]] = 0;
                if (delta > 0) break;   // e is uncut now, the other side is moot
            }
        }
        if (round == 0) break;
    }
    g_netGain += gained;
    return gained;
}

// Grow part 0 from a random seed by best gain until it reaches targ0.
static void greedyGrow(const HGraph &g, vector<signed char> &part, ll maxW0, ll targ0, uint64_t &rs)
{
    const int n = g.n;
    part.assign(n, 1);
    if (n == 0 || targ0 <= 0) return;
    VI pc0(g.m, 0), pc1(g.m), gain(n, 0);
    for (int e = 0; e < g.m; ++e) pc1[e] = g.eptr[e + 1] - g.eptr[e];
    for (int v = 0; v < n; ++v) {
        int gg = 0;
        for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
            int e = g.vind[j];
            if (pc1[e] == 1) gg += g.nw[e];
            gg -= g.nw[e];
        }
        gain[v] = gg;
    }
    Heap h;
    h.init(n, &gain);
    ll w0 = 0;
    h.push((int)(xr(rs) % (uint64_t)n));
    while (w0 < targ0) {
        int v = h.top();
        if (v < 0) break;
        h.pop();
        if (part[v] == 0 || w0 + g.vw[v] > maxW0) continue;
        for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
            int e = g.vind[j];
            int f = pc1[e], t = pc0[e], w = g.nw[e], b = g.eptr[e], en = g.eptr[e + 1];
            bool upd = (en - b) <= FMLIM;
            if (upd && t == 0)
                for (int q = b; q < en; ++q) {
                    int u = g.eind[q];
                    if (u != v && part[u] == 1) { gain[u] += w; h.push(u); }
                }
            --f; ++t;
            if (upd && f == 1)
                for (int q = b; q < en; ++q) {
                    int u = g.eind[q];
                    if (u != v && part[u] == 1) { gain[u] += w; h.push(u); break; }
                }
            pc1[e] = f; pc0[e] = t;
        }
        part[v] = 0;
        w0 += g.vw[v];
    }
    if (w0 < targ0) {
        int off = (int)(xr(rs) % (uint64_t)n);
        for (int i = 0; i < n && w0 < targ0; ++i) {
            int v = off + i;
            if (v >= n) v -= n;
            if (part[v] != 1 || w0 + g.vw[v] > maxW0) continue;
            part[v] = 0;
            w0 += g.vw[v];
        }
    }
}

static vector<signed char> initialPartition(const HGraph &g, ll maxW0, ll maxW1, ll targ0, uint64_t &rs)
{
    int trials = (g.n <= 400) ? 30 : (g.n <= 2000) ? 16 : (g.n <= 8000 ? 6 : 3);
    vector<signed char> best, cur;
    ll bestCut = LLONG_MAX;
    for (int t = 0; t < trials; ++t) {
        greedyGrow(g, cur, maxW0, targ0, rs);
        ll c = fmRefine(g, cur, maxW0, maxW1, 12);
        if (c < bestCut) { bestCut = c; best = cur; }
        if (outOfTime()) break;
    }
    if (best.empty()) best.assign(g.n, 1);
    return best;
}

static ll cutOf(const HGraph &g, const vector<signed char> &part)
{
    ll c = 0;
    for (int e = 0; e < g.m; ++e) {
        int b = g.eptr[e], en = g.eptr[e + 1];
        if (en - b < 2) continue;
        signed char p0 = part[g.eind[b]];
        for (int j = b + 1; j < en; ++j)
            if (part[g.eind[j]] != p0) { c += g.nw[e]; break; }
    }
    return c;
}

// Dinic max-flow on an adjacency-list network.  The augmenting DFS is iterative
// so a deep level graph cannot overflow the stack; maxflow stops early once
// `limit` is reached, which is all a caller testing "can it beat the cut" needs.
struct Dinic {
    VI head, nxt, to, cap, lvl, it, q, path;
    int N = 0;
    void init(int n) { N = n; head.assign(n, -1); nxt.clear(); to.clear(); cap.clear(); }
    void add(int u, int v, int c) {
        to.push_back(v); cap.push_back(c); nxt.push_back(head[u]); head[u] = (int)to.size() - 1;
        to.push_back(u); cap.push_back(0); nxt.push_back(head[v]); head[v] = (int)to.size() - 1;
    }
    // lvl[u] >= 0 iff s reaches u (rev=false) / u reaches s (rev=true) in the residual graph.
    bool reach(int s, int t, bool rev) {
        lvl.assign(N, -1); lvl[s] = 0; q.clear(); q.push_back(s);
        for (size_t i = 0; i < q.size(); ++i) {
            int u = q[i];
            for (int e = head[u]; e >= 0; e = nxt[e]) {
                int v = to[e];
                if ((rev ? cap[e ^ 1] : cap[e]) > 0 && lvl[v] < 0) { lvl[v] = lvl[u] + 1; q.push_back(v); }
            }
        }
        return t >= 0 && lvl[t] >= 0;
    }
    ll maxflow(int s, int t, ll limit) {
        ll fl = 0;
        while (fl < limit && reach(s, t, false)) {
            it = head;
            path.clear();
            int u = s;
            while (fl < limit) {
                if (u == t) {
                    int f = INT_MAX;
                    for (size_t i = 0; i < path.size(); ++i) f = min(f, cap[path[i]]);
                    for (size_t i = 0; i < path.size(); ++i) { cap[path[i]] -= f; cap[path[i] ^ 1] += f; }
                    fl += f; path.clear(); u = s;
                    continue;
                }
                int e = it[u];
                while (e >= 0 && !(cap[e] > 0 && lvl[to[e]] == lvl[u] + 1)) e = nxt[e];
                it[u] = e;
                if (e >= 0) { path.push_back(e); u = to[e]; }
                else if (u == s) break;
                else { lvl[u] = -1; int pe = path.back(); path.pop_back(); u = to[pe ^ 1]; it[u] = nxt[pe]; }
            }
        }
        return fl;
    }
};

// One max-flow min-cut step on the region around the cut.  Each side's region
// is grown by BFS from the boundary up to alpha times the *other* side's slack,
// since in the worst case all of it changes side.  Lawler network: v -> e_in
// (inf), e_in -> e_out (w), e_out -> v (inf); pins outside the region are the
// source/sink.  A net with fixed pins on both sides is cut whatever happens and
// is left out.  Returns the cut reduction; noGain is set when even the
// unconstrained min cut cannot beat the current cut (no smaller region can).
static ll flowRefine(const HGraph &g, vector<signed char> &part, ll maxW0, ll maxW1,
                     int alpha, bool &noGain)
{
    const int n = g.n, m = g.m;
    noGain = true;
    if (n == 0 || m == 0) return 0;
    VI pc0(m, 0), pc1(m, 0), cutNets;
    ll w0 = 0, w1 = 0;
    for (int v = 0; v < n; ++v) { if (part[v]) w1 += g.vw[v]; else w0 += g.vw[v]; }
    for (int e = 0; e < m; ++e) {
        int a = 0, b = 0;
        for (int j = g.eptr[e]; j < g.eptr[e + 1]; ++j) { if (part[g.eind[j]]) ++b; else ++a; }
        pc0[e] = a; pc1[e] = b;
        if (a && b) cutNets.push_back(e);
    }
    if (cutNets.empty()) return 0;
    // Seed with the pins of the small cut nets first: a giant cut net would
    // otherwise fill the region with its own scattered pins.
    stable_sort(cutNets.begin(), cutNets.end(), [&](int a, int b) {
        return g.eptr[a + 1] - g.eptr[a] < g.eptr[b + 1] - g.eptr[b]; });
    ll capB[2] = {(ll)alpha * max(maxW1 - w1, 1LL), (ll)alpha * max(maxW0 - w0, 1LL)};
    VI loc(n, -1), reg;
    int nB[2] = {0, 0};
    ll wB[2] = {0, 0};
    auto tryAdd = [&](int v) {
        int s = part[v];
        if (loc[v] >= 0 || nB[s] >= FLOWCAP || wB[s] + g.vw[v] > capB[s]) return;
        loc[v] = (int)reg.size(); reg.push_back(v); ++nB[s]; wB[s] += g.vw[v];
    };
    for (size_t i = 0; i < cutNets.size(); ++i)
        for (int j = g.eptr[cutNets[i]]; j < g.eptr[cutNets[i] + 1]; ++j) tryAdd(g.eind[j]);
    vector<char> seen(m, 0);
    for (size_t i = 0; i < reg.size(); ++i) {
        int v = reg[i];
        char bit = part[v] ? 2 : 1;
        for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
            int e = g.vind[j];
            if (g.eptr[e + 1] - g.eptr[e] > FLOWEXP || (seen[e] & bit)) continue;
            seen[e] |= bit;
            for (int q = g.eptr[e]; q < g.eptr[e + 1]; ++q)
                if (part[g.eind[q]] == part[v]) tryAdd(g.eind[q]);
        }
    }
    const int R = (int)reg.size(), S = R, T = R + 1;
    if (R == 0) return 0;
    VI tn, r0(m, 0), r1(m, 0), nid(m, -1);
    fill(seen.begin(), seen.end(), 0);
    for (int i = 0; i < R; ++i) {
        int v = reg[i];
        for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
            int e = g.vind[j];
            if (!seen[e]) { seen[e] = 1; tn.push_back(e); }
            if (part[v]) ++r1[e]; else ++r0[e];
        }
    }
    Dinic D;
    D.init(R + 2 + 2 * (int)tn.size());
    int nn = R + 2;
    ll oldCut = 0;
    for (size_t i = 0; i < tn.size(); ++i) {
        int e = tn[i];
        bool f0 = pc0[e] > r0[e], f1 = pc1[e] > r1[e];
        if (f0 && f1) continue;
        if (pc0[e] && pc1[e]) oldCut += g.nw[e];
        nid[e] = nn;
        D.add(nn, nn + 1, g.nw[e]);
        if (f0) D.add(S, nn, INFCAP);
        if (f1) D.add(nn + 1, T, INFCAP);
        nn += 2;
    }
    if (oldCut == 0) return 0;
    for (int i = 0; i < R; ++i) {
        int v = reg[i];
        for (int j = g.vptr[v]; j < g.vptr[v + 1]; ++j) {
            int e = nid[g.vind[j]];
            if (e >= 0) { D.add(i, e, INFCAP); D.add(e + 1, i, INFCAP); }
        }
    }
    if (D.maxflow(S, T, oldCut) >= oldCut) return 0;
    noGain = false;
    ll before = cutOf(g, part);
    vector<signed char> orig(R), side(R);
    for (int i = 0; i < R; ++i) orig[i] = part[reg[i]];
    // The source-side and the sink-side minimum cuts are both minimum; take the
    // first that respects the caps.  A vertex without arcs is unconstrained by
    // the network and keeps its side so it cannot shift the balance for nothing.
    for (int pass = 0; pass < 2; ++pass) {
        D.reach(pass ? T : S, -1, pass == 1);
        ll nw0 = w0;
        for (int i = 0; i < R; ++i) {
            bool r = D.lvl[i] >= 0;
            signed char s = (D.head[i] < 0) ? orig[i] : (signed char)(pass == 0 ? (r ? 0 : 1) : (r ? 1 : 0));
            side[i] = s;
            if (s != orig[i]) nw0 += s ? -(ll)g.vw[reg[i]] : (ll)g.vw[reg[i]];
        }
        if (nw0 > maxW0 || w0 + w1 - nw0 > maxW1) continue;
        for (int i = 0; i < R; ++i) part[reg[i]] = side[i];
        ll after = cutOf(g, part);
        if (after < before) return before - after;
        for (int i = 0; i < R; ++i) part[reg[i]] = orig[i];
    }
    return 0;
}

// Flow steps alternated with FM.  An infeasible min cut halves alpha; a min cut
// that cannot beat the current cut ends the loop.  Returns the flow gain only;
// callers that need the cut re-measure it.
static ll flowLoop(const HGraph &g, vector<signed char> &part, ll maxW0, ll maxW1)
{
    if (g.n < 2 * COARSEST) return 0;
    double t = nowSec();
    ll gained = 0;
    for (int alpha = FLOWALPHA, tries = 0; alpha >= 1 && tries < 12 && !outOfTime(); ++tries) {
        bool noGain = true;
        ll d = flowRefine(g, part, maxW0, maxW1, alpha, noGain);
        ++g_flowRuns;
        if (d > 0) { gained += d; fmRefine(g, part, maxW0, maxW1, 2); }
        else if (noGain) break;
        else alpha >>= 1;
    }
    g_flowGain += gained;
    g_tFlow += nowSec() - t;
    return gained;
}

// One multilevel pass.  scratch=false is a V-cycle: clustering may not merge
// vertices of different blocks, so the projected coarse partition has exactly
// the cut of `part` and every fmRefine below can only improve on it.
static ll mlPass(const HGraph &g0, vector<signed char> &part, bool scratch,
                 ll maxW0, ll maxW1, ll targ0, ll maxNodeW, uint64_t &rs, bool hec)
{
    vector<HGraph> lv;
    vector<VI> cm;
    vector<vector<signed char>> cparts;
    lv.reserve(MAXLV); cm.reserve(MAXLV); cparts.reserve(MAXLV);
    const HGraph *cur = &g0;
    const signed char *restr = scratch ? nullptr : part.data();
    double tc = nowSec();
    while (cur->n > COARSEST && (int)lv.size() < MAXLV && !outOfTime()) {
        VI cmap;
        int nc = clusterLevel(*cur, cmap, maxNodeW, rs, restr, hec);
        if (nc < 2 || nc > (int)(0.92 * cur->n)) break;
        HGraph cg;
        contract(*cur, cmap, nc, cg);
        cg.buildInc();
        if (!scratch) {
            vector<signed char> cp(nc, 0);
            for (int v = 0; v < cur->n; ++v) cp[cmap[v]] = restr[v];
            cparts.push_back(move(cp));
            restr = cparts.back().data();
        }
        lv.push_back(move(cg));
        cm.push_back(move(cmap));
        cur = &lv.back();
        ++g_levels;
        if (cur->m == 0) break;
    }
    g_tCoarsen += nowSec() - tc;

    double ti = nowSec();
    vector<signed char> cp;
    if (scratch) {
        cp = initialPartition(*cur, maxW0, maxW1, targ0, rs);
    } else {
        cp = cparts.empty() ? part : cparts.back();
        fmRefine(*cur, cp, maxW0, maxW1, 8);
    }
    g_tInit += nowSec() - ti;

    double tr = nowSec();
    for (int i = (int)lv.size() - 1; i >= 0; --i) {
        const HGraph &fine = (i == 0) ? g0 : lv[i - 1];
        vector<signed char> fp(fine.n);
        const VI &mp = cm[i];
        for (int v = 0; v < fine.n; ++v) fp[v] = cp[mp[v]];
        cp.swap(fp);
        fmRefine(fine, cp, maxW0, maxW1, UPASS);
        // Compound moves need balance slack to fit, so they are only worth
        // trying where the vertices are still light relative to the cap.
        if ((i == 0 || fine.n >= 20000) && !outOfTime()) {
            ll ng = netMoveRefine(fine, cp, maxW0, maxW1, 2);
            if (ng > 0 && i == 0) fmRefine(fine, cp, maxW0, maxW1, 3);
        }
    }
    g_tRefine += nowSec() - tr;
    part.swap(cp);
    return cutOf(g0, part);
}

// Multilevel restarts from scratch (best kept), then V-cycles on the winner.
// The first minR restarts always run untruncated; further ones borrow
// g_deadline = tEnd so a late restart cannot overrun the local budget.
static vector<signed char> bisect(const HGraph &g0, ll maxW0, ll maxW1, ll targ0,
                                  uint64_t seedv, int minR, double tEnd)
{
    ++g_bisections;
    vector<signed char> best, part;
    if (g0.n == 0) return best;
    // A net whose pins outweigh both caps can never be uncut: it adds the same
    // constant to every candidate, so dropping it is exact, and it stops a giant
    // net from marking half the vertices as FM boundary in every pass.
    HGraph gf;
    const HGraph *gp = &g0;
    {
        ll capMax = max(maxW0, maxW1);
        int drop = 0;
        for (int e = 0; e < g0.m; ++e) {
            ll s = 0;
            for (int j = g0.eptr[e]; j < g0.eptr[e + 1]; ++j) s += g0.vw[g0.eind[j]];
            if (s > capMax) ++drop;
        }
        if (drop > 0) {
            gf.n = g0.n;
            gf.vw = g0.vw;
            gf.eptr.push_back(0);
            gf.eind.reserve(g0.eind.size());
            for (int e = 0; e < g0.m; ++e) {
                ll s = 0;
                for (int j = g0.eptr[e]; j < g0.eptr[e + 1]; ++j) s += g0.vw[g0.eind[j]];
                if (s > capMax) continue;
                for (int j = g0.eptr[e]; j < g0.eptr[e + 1]; ++j) gf.eind.push_back(g0.eind[j]);
                gf.eptr.push_back((int)gf.eind.size());
                gf.nw.push_back(g0.nw[e]);
            }
            gf.m = (int)gf.nw.size();
            gf.buildInc();
            gp = &gf;
        }
    }
    const HGraph &g = *gp;
    ll slack = maxW0 + maxW1 - g.totW;
    if (slack < 1) slack = 1;
    ll maxNodeW = (ll)ceil(1.5 * (double)g.totW / (double)COARSEST);
    if (maxNodeW < 1) maxNodeW = 1;
    if (maxNodeW > slack) maxNodeW = slack;   // keeps a balanced initial bisection reachable
    ll bestCut = LLONG_MAX;
    const double hardDl = g_deadline;
    if (tEnd > hardDl) tEnd = hardDl;
    double tStart = nowSec();
    bool bestHec = false;
    // Restarts stop once `pat` of them in a row fail to beat the best bisection.
    // A restart is milliseconds on a small graph and seconds on a Titan one, so
    // the big graphs (where we run at a fraction of KaHyPar's time already and
    // the 20th restart never wins) give up much earlier than the small ones.
    int lastImp = 0, pat = (g.n <= 20000) ? 12 : 5;
    for (int r = 0; r < RMAX; ++r) {
        ++g_restarts;
        bool hec = g_useHEC && (r % 2 == 1);   // both coarsening flavours, best cut wins
        if (hec) ++g_hecTries;
        g_deadline = (r < minR) ? hardDl : tEnd;
        uint64_t rs = seedv * 1000003ULL + (uint64_t)r * 7919ULL + 88172645463325252ULL;
        part.assign(g.n, 1);
        ll c = mlPass(g, part, true, maxW0, maxW1, targ0, maxNodeW, rs, hec);
        if (c < bestCut) { bestCut = c; bestHec = hec; best.swap(part); lastImp = r; }
        double now = nowSec();
        if (now > hardDl) break;
        if (r + 1 >= minR && r - lastImp >= pat) break;
        if (r + 1 >= minR && now + 1.1 * (now - tStart) / (double)(r + 1) > tEnd) break;
    }
    g_deadline = hardDl;
    if (bestHec) ++g_hecWins;
    if (best.empty()) best.assign(g.n, 1);
    if (flowLoop(g, best, maxW0, maxW1) > 0) bestCut = cutOf(g, best);
    uint64_t rs = seedv * 2246822519ULL + 7ULL;
    int fails = 0;
    for (int it = 0; it < 8 && !outOfTime(); ++it) {
        if (it >= 3 && nowSec() > tEnd) break;
        double vs0 = nowSec();
        part = best;
        ll c = mlPass(g, part, false, maxW0, maxW1, targ0, maxNodeW, rs, false);
        ++g_vcyc;
        if (c < bestCut) { bestCut = c; best.swap(part); fails = 0; continue; }
        // Clustering is randomised, so one failed V-cycle is not a fixed point;
        // retry only while another whole cycle still fits the local budget.
        if (++fails >= 2 || nowSec() + (nowSec() - vs0) > tEnd) break;
    }
    flowLoop(g, best, maxW0, maxW1);
    return best;
}

static void extract(const HGraph &g, const vector<signed char> &part, int side,
                    const VI &gid, HGraph &sg, VI &sgid)
{
    VI mp(g.n, -1);
    sgid.clear();
    for (int v = 0; v < g.n; ++v)
        if (part[v] == side) { mp[v] = (int)sgid.size(); sgid.push_back(gid[v]); }
    sg.n = (int)sgid.size();
    sg.vw.assign(sg.n, 1);
    for (int v = 0, i = 0; v < g.n; ++v) if (part[v] == side) sg.vw[i++] = g.vw[v];
    sg.eptr.clear(); sg.eind.clear(); sg.nw.clear();
    sg.eptr.push_back(0);
    for (int e = 0; e < g.m; ++e) {
        int b = g.eptr[e], en = g.eptr[e + 1];
        if (en - b < 2) continue;
        bool all = true;
        for (int j = b; j < en; ++j) if (part[g.eind[j]] != side) { all = false; break; }
        if (!all) continue;
        for (int j = b; j < en; ++j) sg.eind.push_back(mp[g.eind[j]]);
        sg.eptr.push_back((int)sg.eind.size());
        sg.nw.push_back(g.nw[e]);
    }
    sg.m = (int)sg.nw.size();
    sg.buildInc();
}

static inline int lg2c(int k) { int d = 0; while ((1 << d) < k) ++d; return d; }

static void recurse(const HGraph &g, const VI &gid, int kparts, int off, ll L, VI &out,
                    uint64_t seedv, double tbudget)
{
    if (kparts <= 1) {
        for (int v = 0; v < g.n; ++v) out[gid[v]] = off;
        return;
    }
    int k0 = kparts / 2, k1 = kparts - k0;
    ll W = g.totW;
    ll ideal0 = (ll)((double)W * k0 / kparts + 0.5);
    // A side that is already a final block may use its whole slack; only a side
    // that still has to be split must reserve some for the bisections below it.
    double a0 = (k0 == 1) ? 1.0 : 0.7, a1 = (k1 == 1) ? 1.0 : 0.7;
    ll maxW0 = ideal0 + (ll)floor(a0 * ((double)k0 * L - (double)ideal0));
    ll ideal1 = W - ideal0;
    ll maxW1 = ideal1 + (ll)floor(a1 * ((double)k1 * L - (double)ideal1));
    if (maxW0 > (ll)k0 * L) maxW0 = (ll)k0 * L;
    if (maxW1 > (ll)k1 * L) maxW1 = (ll)k1 * L;
    if (ideal0 > maxW0) ideal0 = maxW0;
    if (W - ideal0 > maxW1) ideal0 = W - maxW1;
    if (ideal0 < 0) ideal0 = 0;
    int minR = (g.n <= 5000) ? 8 : (g.n <= 20000) ? 6 : (g.n <= 60000) ? 5 : (g.n <= 200000) ? 4 : 3;
    int D = max(1, lg2c(kparts));
    double tb0 = nowSec();
    vector<signed char> part = bisect(g, maxW0, maxW1, ideal0, seedv, minR, tb0 + tbudget / D);
    if (kparts == 2) {
        for (int v = 0; v < g.n; ++v) out[gid[v]] = off + (int)part[v];
        return;
    }
    double rest = max(0.0, tbudget - (nowSec() - tb0));
    HGraph s0, s1;
    VI id0, id1;
    extract(g, part, 0, gid, s0, id0);
    extract(g, part, 1, gid, s1, id1);
    part.clear();
    part.shrink_to_fit();
    double w0 = (double)s0.eind.size() * lg2c(k0), w1 = (double)s1.eind.size() * lg2c(k1);
    double b0 = (w0 + w1 > 0.0) ? rest * w0 / (w0 + w1) : 0.5 * rest;
    recurse(s0, id0, k0, off, L, out, seedv * 31 + 1, b0);
    recurse(s1, id1, k1, off + k0, L, out, seedv * 31 + 2, rest - b0);
}

// Polish the k-way result pair by pair.  A net with a pin outside the pair has
// a pin in a third block whatever we do inside the pair, so restricting the
// sub-problem to nets wholly inside the pair is exact for the cut-net metric.
// level 0: flat FM + compound moves + flows; level>0: a restricted V-cycle on
// the pair; level 2 also re-bisects the pair from scratch, which is the only
// move that can revisit the top split (recursive bisection fixes it while
// optimising a different net set than the pair is finally judged by).  A pair
// is committed only when its own cut strictly drops and both caps still hold.
static void pairRefine(const HGraph &g, VI &part, int k, ll L, int level,
                       uint64_t seedv, double tEnd)
{
    if (k < 3) return;
    VI mp(g.n, -1), ids;
    vector<signed char> sub, alt;
    HGraph sg;
    uint64_t rs = seedv * 2654435761ULL + 88172645463325252ULL;
    const double hardDl = g_deadline;
    if (tEnd > hardDl) tEnd = hardDl;
    int left = k * (k - 1) / 2;
    for (int round = 0, rmax = level ? 1 : 2; round < rmax; ++round) {
        bool improved = false;
        for (int a = 0; a < k; ++a) for (int b = a + 1; b < k; ++b) {
            if (nowSec() > tEnd) { g_deadline = hardDl; return; }
            --left;
            ids.clear();
            for (int v = 0; v < g.n; ++v)
                if (part[v] == a || part[v] == b) { mp[v] = (int)ids.size(); ids.push_back(v); }
            if (ids.size() < 2) continue;
            sg.n = (int)ids.size();
            sg.vw.resize(sg.n);
            for (int i = 0; i < sg.n; ++i) sg.vw[i] = g.vw[ids[i]];
            sg.eptr.clear(); sg.eind.clear(); sg.nw.clear();
            sg.eptr.push_back(0);
            for (int e = 0; e < g.m; ++e) {
                int bg = g.eptr[e], en = g.eptr[e + 1];
                if (en - bg < 2) continue;
                bool ok = true;
                for (int j = bg; j < en; ++j) { int p = part[g.eind[j]]; if (p != a && p != b) { ok = false; break; } }
                if (!ok) continue;
                for (int j = bg; j < en; ++j) sg.eind.push_back(mp[g.eind[j]]);
                sg.eptr.push_back((int)sg.eind.size());
                sg.nw.push_back(g.nw[e]);
            }
            sg.m = (int)sg.nw.size();
            if (sg.m == 0) continue;
            sg.buildInc();
            sub.assign(sg.n, 0);
            for (int i = 0; i < sg.n; ++i) sub[i] = (part[ids[i]] == b) ? 1 : 0;
            ll before = cutOf(sg, sub), after;
            if (level && sg.n > 4 * COARSEST) {
                // Each remaining pair gets an equal share of what is left of the
                // sweep, and g_deadline is borrowed so one pair cannot overrun it.
                // The cluster weight must stay inside the pair's balance slack or
                // the coarse FM cannot move anything.
                double now = nowSec();
                double pEnd = min(tEnd, now + (tEnd - now) / (double)max(1, left + 1));
                ll slack = 2 * L - sg.totW;
                if (slack < 1) slack = 1;
                ll mnw = (ll)ceil(1.5 * (double)sg.totW / (double)COARSEST);
                if (mnw < 1) mnw = 1;
                if (mnw > slack) mnw = slack;
                g_deadline = pEnd;
                after = mlPass(sg, sub, false, L, L, sg.totW / 2, mnw, rs, false);
                if (flowLoop(sg, sub, L, L) > 0) after = cutOf(sg, sub);
                ++g_deepPairs;
                if (level >= 2 && nowSec() < pEnd) {
                    alt = bisect(sg, L, L, sg.totW / 2, rs + 7919ULL, 1, pEnd);
                    if ((int)alt.size() == sg.n) {
                        ll wa = 0;
                        for (int i = 0; i < sg.n; ++i) if (!alt[i]) wa += sg.vw[i];
                        if (wa <= L && sg.totW - wa <= L) {   // a truncated bisect may be infeasible
                            ll ca = cutOf(sg, alt);
                            if (ca < after) { after = ca; sub.swap(alt); ++g_scratchWins; }
                        }
                    }
                }
                g_deadline = hardDl;
            } else {
                after = fmRefine(sg, sub, L, L, 6);
                after -= netMoveRefine(sg, sub, L, L, 2);
                if (flowLoop(sg, sub, L, L) > 0) after = cutOf(sg, sub);
            }
            if (after < before) {
                improved = true;
                g_pairGain += (int)(before - after);
                for (int i = 0; i < sg.n; ++i) part[ids[i]] = sub[i] ? b : a;
            }
        }
        if (!improved) break;
    }
    g_deadline = hardDl;
}

int main(int argc, char **argv)
{
    double t0 = nowSec();
    string input, out;
    int k = 2;
    double eps = 0.02;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a.rfind("--k=", 0) == 0) k = atoi(a.c_str() + 4);
        else if (a.rfind("--eps=", 0) == 0) eps = atof(a.c_str() + 6);
        else if (a.rfind("--seed=", 0) == 0) seed = (unsigned)strtoul(a.c_str() + 7, nullptr, 10);
        else if (a.rfind("--out=", 0) == 0) out = a.substr(6);
        else if (a[0] != '-') input = a;
    }
    if (input.empty() || out.empty() || k < 2) {
        fprintf(stderr, "usage: %s <input.hgr> --k=K --eps=EPS --seed=S --out=FILE\n", argv[0]);
        return 2;
    }
    HGraph g;
    if (!readHgr(input.c_str(), g)) { fprintf(stderr, "error: cannot read %s\n", input.c_str()); return 1; }
    double tread = nowSec() - t0;
    const int n = g.n;
    ll L = (ll)floor((1.0 + eps) * (double)((n + k - 1) / k));
    if (L < 1) L = 1;
    // Effort is budgeted against a crude model of KaHyPar's own runtime, which
    // is dominated by pins in large nets; T is our wall-clock target, the
    // fixed minimum restart schedule in recurse() applies regardless.
    size_t bigPins = 0;
    int maxNet = 0, nBig = 0, nHuge = 0;
    for (int e = 0; e < g.m; ++e) {
        int sz = g.eptr[e + 1] - g.eptr[e];
        if (sz > 60) { bigPins += sz; ++nBig; }
        if (sz > MOVELIM) ++nHuge;
        if (sz > maxNet) maxNet = sz;
    }
    double pins = (double)g.eind.size();
    double bigShare = pins > 0.0 ? (double)bigPins / pins : 0.0;
    double tK = pins * sqrt(0.5 * k) * (1.0e-5 + 3.2e-4 * bigShare);
    double T = min(240.0, max(1.2, 0.22 * tK));
    g_deadline = t0 + max(20.0, min(1100.0, 4.0 * T + 8.0 + 4e-5 * pins * k));
    // Whole-net contraction only pays where large nets hold a real share of the
    // pins; on small-net designs the pairwise rating already sees every net.
    g_useHEC = (bigShare >= 0.12);

    auto cutK = [&](const VI &p) {
        ll c = 0;
        for (int e = 0; e < g.m; ++e) {
            int b = g.eptr[e], en = g.eptr[e + 1];
            if (en - b < 2) continue;
            int p0 = p[g.eind[b]];
            for (int j = b + 1; j < en; ++j)
                if (p[g.eind[j]] != p0) { ++c; break; }
        }
        return c;
    };
    VI outPart(n, 0), gid(n);
    for (int v = 0; v < n; ++v) gid[v] = v;
    // Restarts inside a bisection pick the best *bisection*, which is not the
    // best k-way result; for k>=3 a second full RB run, scored on the real
    // k-way cut, is the only way to see that.  Further reps buy less than the
    // pair sweep below, which reworks 2/k of the graph at a time.
    ll bestCut = LLONG_MAX;
    int reps = (k >= 3) ? 2 : 1;
    int deepSweeps = 0;
    VI cur(n, 0), scnt(k, 0);
    auto feasible = [&](const VI &p) {
        fill(scnt.begin(), scnt.end(), 0);
        for (int v = 0; v < n; ++v) {
            if (p[v] < 0 || p[v] >= k) return false;
            ++scnt[p[v]];
        }
        for (int q = 0; q < k; ++q) if (scnt[q] > L) return false;
        return true;
    };
    double tRB = nowSec();
    for (int rep = 0; rep < reps; ++rep) {
        double r0 = nowSec();
        recurse(g, gid, k, 0, L, cur, (uint64_t)seed * 2654435761ULL + 1 + (uint64_t)rep * 7919ULL,
                0.8 * T / reps);
        double tp = nowSec();
        pairRefine(g, cur, k, L, 0, (uint64_t)seed + rep, g_deadline);
        g_tPair += nowSec() - tp;
        ++g_runs;
        if (feasible(cur)) {
            ll c = cutK(cur);
            if (c < bestCut) { bestCut = c; outPart = cur; }
        }
        if (nowSec() + (nowSec() - r0) > g_deadline) break;
    }
    if (bestCut == LLONG_MAX) outPart = cur;   // nothing feasible: repair pass below
    else if (k >= 3) {
        // Re-partition sweeps over the block pairs of the winner, repeated inside
        // one budget window while they keep paying.  A committed pair changes the
        // net set every neighbouring pair is judged by (a net with a pin in block
        // c enters pair (a,b) only once c's contents move), so a second sweep is
        // a different problem, not a fixed point.  Budgeted against the effort
        // already spent and against tK, the estimate of KaHyPar's runtime: where
        // we are far ahead of it the sweeps are nearly free.  Each sweep works on
        // a copy that is re-checked, so the loop can never cost the cell.
        double tp = nowSec();
        double spent = tp - tRB;
        double budget = max(0.30 * spent, min(0.8 * T, tK - spent));
        budget = min(budget, spent);
        budget = min(budget, max(0.0, g_deadline - tp));
        double sweepEnd = tp + budget;
        for (int sweep = 0; sweep < SWEEPS && nowSec() < sweepEnd - 0.01; ++sweep) {
            VI trial(outPart);
            pairRefine(g, trial, k, L, 2,
                       (uint64_t)seed * 977 + 13 + (uint64_t)sweep * 131, sweepEnd);
            if (!feasible(trial)) break;
            ll c = cutK(trial);
            if (c >= bestCut) break;
            bestCut = c;
            outPart.swap(trial);
            ++deepSweeps;
        }
        g_tPair += nowSec() - tp;
    }

    VI cnt(k, 0);
    for (int v = 0; v < n; ++v) {
        if (outPart[v] < 0 || outPart[v] >= k) outPart[v] = 0;
        ++cnt[outPart[v]];
    }
    for (int p = 0; p < k; ++p) {                 // safety net; should never fire
        while (cnt[p] > L) {
            int q = -1;
            for (int t = 0; t < k; ++t) if (cnt[t] < L) { q = t; break; }
            if (q < 0) break;
            int moved = -1;
            for (int v = 0; v < n; ++v) if (outPart[v] == p) { outPart[v] = q; moved = v; break; }
            if (moved < 0) break;
            --cnt[p]; ++cnt[q];
        }
    }

    ll cut = cutK(outPart);
    ll mx = 0;
    for (int p = 0; p < k; ++p) mx = max(mx, (ll)cnt[p]);

    string buf;
    buf.reserve((size_t)n * 3);
    char tmp[16];
    for (int v = 0; v < n; ++v) {
        int p = outPart[v];
        if (p < 10) buf += (char)('0' + p);
        else { int l = snprintf(tmp, sizeof(tmp), "%d", p); buf.append(tmp, l); }
        buf += '\n';
    }
    FILE *f = fopen(out.c_str(), "wb");
    if (!f) { fprintf(stderr, "error: cannot write %s\n", out.c_str()); return 1; }
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);

    double tot = nowSec() - t0;
    printf("n=%d m=%d pins=%zu k=%d cap=%lld maxpart=%lld\n", n, g.m, g.eind.size(), k, L, mx);
    printf("time total=%.2f read=%.2f coarsen=%.2f init=%.2f refine=%.2f pair=%.2f flow=%.2f T=%.1f hard=%.1f bigshare=%.2f\n",
           tot, tread, g_tCoarsen, g_tInit, g_tRefine, g_tPair, g_tFlow, T, g_deadline - t0, bigShare);
    printf("levels=%d bisections=%d restarts=%d vcycles=%d pairgain=%d rbruns=%d deep=%d sweeps=%d scratchwin=%d flowruns=%d flowgain=%lld\n",
           g_levels, g_bisections, g_restarts, g_vcyc, g_pairGain, g_runs, g_deepPairs, deepSweeps,
           g_scratchWins, g_flowRuns, g_flowGain);
    printf("nets>60=%d nets>%d=%d maxnet=%d netmoves=%lld netgain=%lld hecwin=%d/%d\n",
           nBig, MOVELIM, nHuge, maxNet, g_netMoves, g_netGain, g_hecWins, g_hecTries);
    ll cb[3] = {0, 0, 0}, tbk[3] = {0, 0, 0};
    for (int e = 0; e < g.m; ++e) {
        int b = g.eptr[e], en = g.eptr[e + 1];
        if (en - b < 2) continue;
        int bk = (en - b <= 6) ? 0 : ((en - b <= 60) ? 1 : 2);
        ++tbk[bk];
        int p0 = outPart[g.eind[b]];
        for (int j = b + 1; j < en; ++j)
            if (outPart[g.eind[j]] != p0) { ++cb[bk]; break; }
    }
    printf("cutbysize <=6:%lld/%lld 7..60:%lld/%lld >60:%lld/%lld\n",
           cb[0], tbk[0], cb[1], tbk[1], cb[2], tbk[2]);
    printf("RESULT cut=%lld feasible=%d\n", cut, mx <= L ? 1 : 0);
    return 0;
}
