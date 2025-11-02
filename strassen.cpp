// strassen.cpp
// Depth-2 Strassen (7x7 = 49 leaves), one leaf per rank.
// Two pinned input blocks per rank (from placement), fetch the rest (typically 4–6).
// Each leaf computes a single 2x2 product and contributes to specific C blocks.
//
// Compile: mpicxx -O2 -std=c++17 strassen.cpp -o strassen
// Run:     mpirun -np 49 ./strassen
//
#include <mpi.h>
#include <array>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <unordered_map>
#include <deque>

using std::string;

// ---------- Types & helpers ----------
struct M2 { double v[2][2]{}; };

static inline M2 add2(const M2& A, const M2& B, double a=1.0, double b=1.0){
    M2 R{};
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
            R.v[i][j] = a*A.v[i][j] + b*B.v[i][j];
    return R;
}
static inline M2 mul2(const M2& A, const M2& B){
    M2 R{};
    R.v[0][0] = A.v[0][0]*B.v[0][0] + A.v[0][1]*B.v[1][0];
    R.v[0][1] = A.v[0][0]*B.v[0][1] + A.v[0][1]*B.v[1][1];
    R.v[1][0] = A.v[1][0]*B.v[0][0] + A.v[1][1]*B.v[1][0];
    R.v[1][1] = A.v[1][0]*B.v[0][1] + A.v[1][1]*B.v[1][1];
    return R;
}
static inline void pack_M2(const M2& m, double buf[4]){
    buf[0]=m.v[0][0]; buf[1]=m.v[0][1];
    buf[2]=m.v[1][0]; buf[3]=m.v[1][1];
}
static inline M2 unpack_M2(const double buf[4]){
    M2 m{};
    m.v[0][0]=buf[0]; m.v[0][1]=buf[1];
    m.v[1][0]=buf[2]; m.v[1][1]=buf[3];
    return m;
}
static inline void stitch8(double A[8][8], const std::array<M2,16>& blocks){
    auto idx = [&](int r2,int c2){ return r2*4 + c2; };
    for(int r2=0;r2<4;r2++){
        for(int c2=0;c2<4;c2++){
            const M2& b = blocks[idx(r2,c2)];
            for(int i=0;i<2;i++)
                for(int j=0;j<2;j++)
                    A[r2*2+i][c2*2+j] = b.v[i][j];
        }
    }
}

// ---------- Leaf recipe representation ----------
struct Term { int id; int8_t s; };     // s ∈ {+1,-1}
struct Leaf {
    std::vector<Term> Aterms;          // ids in [0..15]
    std::vector<Term> Bterms;          // ids in [16..31]
    std::vector<Term> Couts;           // C ids in [0..15] with sign
};

// ---------- Owner store and placement ----------
struct Store { std::map<int,M2> data; };
static std::array<int,32> OWNER;
static inline int owner_of(int id, int /*world*/) { return OWNER[id]; }

// Parse e.g. "A:a11", "B:b33"
static inline int parse_token_to_id(const std::string& tok){
    char AB='A';
    int r=1, c=1;
    if (!tok.empty()) AB = (tok[0]=='A' || tok[0]=='B') ? tok[0] : 'A';
    size_t colon = tok.find(':'); std::string body = (colon==std::string::npos)? tok : tok.substr(colon+1);
    std::vector<int> digs; int acc=-1; bool in=false;
    for(char ch: body){
        if (ch>='0' && ch<='9'){ if(!in){ acc=ch-'0'; in=true;} else acc = acc*10 + (ch-'0'); }
        else { if(in){ digs.push_back(acc); in=false; } }
    }
    if (in) digs.push_back(acc);
    if (digs.size()>=2){ r=digs[0]; c=digs[1]; }
    r = std::max(1,std::min(4,r));
    c = std::max(1,std::min(4,c));
    int base = (AB=='A')?0:16;
    return base + (r-1)*4 + (c-1);
}
static void build_owner_from_placement(const std::vector<std::pair<std::string,std::string>>& placement, int world){
    for (int id=0; id<32; ++id) OWNER[id] = id % world;   // default
    for (int r=0; r<(int)placement.size(); ++r){
        int id1 = parse_token_to_id(placement[r].first);
        int id2 = parse_token_to_id(placement[r].second);
        OWNER[id1] = r; OWNER[id2] = r;
    }
}

// ---------- Non-blocking P2P fetch ----------
static constexpr int TAG_FETCH_REQ  = 100;
static constexpr int TAG_FETCH_DATA = 101;

static inline void owner_serve_once(Store& store){
    while (true){
        int flag=0; MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_FETCH_REQ, MPI_COMM_WORLD, &flag, &st);
        if (!flag) break;
        int req_id = -1;
        MPI_Recv(&req_id, 1, MPI_INT, st.MPI_SOURCE, TAG_FETCH_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double buf[4]{0,0,0,0};
        auto it = store.data.find(req_id);
        if (it != store.data.end()) pack_M2(it->second, buf);
        MPI_Send(buf, 4, MPI_DOUBLE, st.MPI_SOURCE, TAG_FETCH_DATA, MPI_COMM_WORLD);
    }
}
struct PendingFetch {
    int id=-1, owner=-1;
    MPI_Request rq_send = MPI_REQUEST_NULL;
    MPI_Request rq_recv = MPI_REQUEST_NULL;
    double buf[4]{};
    bool send_done=false;
};
template<class OnRecv>
static inline void fetch_many_progress(const std::vector<int>& ids,
                                       int me, int world,
                                       Store& store,
                                       OnRecv on_recv,
                                       unsigned long long& bytes_recv_payload)
{
    std::deque<PendingFetch> pend; pend.resize(ids.size());
    for (size_t i=0;i<ids.size();++i){
        pend[i].id = ids[i];
        pend[i].owner = owner_of(ids[i], world);
        auto it = store.data.find(ids[i]);
        if (it != store.data.end()){
            on_recv(ids[i], it->second);
            std::swap(pend[i], pend.back()); pend.pop_back(); --i;
            continue;
        }
        MPI_Isend(&pend[i].id, 1, MPI_INT, pend[i].owner, TAG_FETCH_REQ,  MPI_COMM_WORLD, &pend[i].rq_send);
        MPI_Irecv(pend[i].buf, 4, MPI_DOUBLE, pend[i].owner, TAG_FETCH_DATA, MPI_COMM_WORLD, &pend[i].rq_recv);
    }
    while(!pend.empty()){
        owner_serve_once(store);
        size_t scan = std::min<size_t>(pend.size(), 8);
        for (size_t k=0;k<scan;++k){
            PendingFetch& p = pend[k];
            if (!p.send_done){ int sdone=0; MPI_Test(&p.rq_send, &sdone, MPI_STATUS_IGNORE); p.send_done = (sdone!=0); }
            int rdone=0; MPI_Test(&p.rq_recv, &rdone, MPI_STATUS_IGNORE);
            if (rdone){
                M2 m = unpack_M2(p.buf);
                on_recv(p.id, m);
                if (p.owner != me) bytes_recv_payload += sizeof(double)*4;
                std::swap(pend[k], pend.back()); pend.pop_back();
                break;
            }
        }
    }
    for (int i=0;i<8;i++) owner_serve_once(store);
}

// ---------- Demo data ----------
static inline M2 make_block(int seed){
    M2 m{};
    m.v[0][0] = seed+1; m.v[0][1] = 0.1*seed;
    m.v[1][0] = 0.1*seed; m.v[1][1] = seed+2;
    return m;
}

// ---------- Depth-2 Strassen leaf generator ----------
// We build 49 leaves directly by composing two Strassen levels symbolically.

using Lin = std::map<int,int>; // id -> coeff (integers)

// Add src into dst with multiplier (+1/-1)
static inline void lin_accum(Lin& dst, const Lin& src, int s){
    for (auto& kv: src) dst[kv.first] += s * kv.second;
}
// Create a singleton Lin (one base block id with coeff +1)
static inline Lin singleton(int id){ return Lin{{id,1}}; }

// Build top-level quadrants: 4 quadrants (q=0..3), each a 2x2 of Lin.
// For A: base ids 0..15; for B: base ids 16..31.
struct Lin2x2 { Lin e[2][2]; }; // 2x2 matrix of linear forms over base IDs

static inline void build_quadrants_A(std::array<Lin2x2,4>& Qa){
    // global 4x4 block ids: id = r*4 + c, r,c in [0..3]
    auto at = [&](int r,int c)->Lin{ return singleton(r*4 + c); };
    // q0: rows 0,1; cols 0,1
    Qa[0].e[0][0]=at(0,0); Qa[0].e[0][1]=at(0,1);
    Qa[0].e[1][0]=at(1,0); Qa[0].e[1][1]=at(1,1);
    // q1: rows 0,1; cols 2,3
    Qa[1].e[0][0]=at(0,2); Qa[1].e[0][1]=at(0,3);
    Qa[1].e[1][0]=at(1,2); Qa[1].e[1][1]=at(1,3);
    // q2: rows 2,3; cols 0,1
    Qa[2].e[0][0]=at(2,0); Qa[2].e[0][1]=at(2,1);
    Qa[2].e[1][0]=at(3,0); Qa[2].e[1][1]=at(3,1);
    // q3: rows 2,3; cols 2,3
    Qa[3].e[0][0]=at(2,2); Qa[3].e[0][1]=at(2,3);
    Qa[3].e[1][0]=at(3,2); Qa[3].e[1][1]=at(3,3);
}
static inline void build_quadrants_B(std::array<Lin2x2,4>& Qb){
    auto bt = [&](int r,int c)->Lin{ return singleton(16 + r*4 + c); };
    Qb[0].e[0][0]=bt(0,0); Qb[0].e[0][1]=bt(0,1);
    Qb[0].e[1][0]=bt(1,0); Qb[0].e[1][1]=bt(1,1);
    Qb[1].e[0][0]=bt(0,2); Qb[1].e[0][1]=bt(0,3);
    Qb[1].e[1][0]=bt(1,2); Qb[1].e[1][1]=bt(1,3);
    Qb[2].e[0][0]=bt(2,0); Qb[2].e[0][1]=bt(2,1);
    Qb[2].e[1][0]=bt(3,0); Qb[2].e[1][1]=bt(3,1);
    Qb[3].e[0][0]=bt(2,2); Qb[3].e[0][1]=bt(2,3);
    Qb[3].e[1][0]=bt(3,2); Qb[3].e[1][1]=bt(3,3);
}

// Combine quadrants with signs: result X = sum_i s_i * Q[q_i]
static inline Lin2x2 combine_quads(const std::array<Lin2x2,4>& Q, const std::vector<std::pair<int,int>>& parts){
    Lin2x2 X;
    for (auto [qi,sg]: parts){
        for (int i=0;i<2;i++) for (int j=0;j<2;j++) lin_accum(X.e[i][j], Q[qi].e[i][j], sg);
    }
    return X;
}

// Second-level Strassen on X(2x2), Y(2x2).
// Returns 7 sub-leaves; for each, yields left Lin (Aterms), right Lin (Bterms), and
// a list of sub-quadrant contributions (s in {0,1,2,3} with sign).
struct SubLeaf { Lin LA, LB; std::vector<std::pair<int,int>> subC; }; // subC: (subpos, sign)
static inline void make_sub_leaves(const Lin2x2& X, const Lin2x2& Y, std::array<SubLeaf,7>& out){
    // m1: (x00+x11)*(y00+y11) -> Csub: 11 and 22 (+1 each)
    out[0].LA = X.e[0][0]; lin_accum(out[0].LA, X.e[1][1], +1);
    out[0].LB = Y.e[0][0]; lin_accum(out[0].LB, Y.e[1][1], +1);
    out[0].subC = {{0, +1}, {3, +1}}; // (0,0)->idx0; (1,1)->idx3

    // m2: (x10+x11)*y00 -> Csub: 21 (+1), 22 (-1)
    out[1].LA = X.e[1][0]; lin_accum(out[1].LA, X.e[1][1], +1);
    out[1].LB = Y.e[0][0];
    out[1].subC = {{2, +1}, {3, -1}};

    // m3: x00*(y01 - y11) -> Csub: 12 (+1), 22 (+1)
    out[2].LA = X.e[0][0];
    out[2].LB = Y.e[0][1]; lin_accum(out[2].LB, Y.e[1][1], -1);
    out[2].subC = {{1, +1}, {3, +1}};

    // m4: x11*(y10 - y00) -> Csub: 11 (+1), 21 (+1)
    out[3].LA = X.e[1][1];
    out[3].LB = Y.e[1][0]; lin_accum(out[3].LB, Y.e[0][0], -1);
    out[3].subC = {{0, +1}, {2, +1}};

    // m5: (x00+x01)*y11 -> Csub: 11 (-1), 12 (+1)
    out[4].LA = X.e[0][0]; lin_accum(out[4].LA, X.e[0][1], +1);
    out[4].LB = Y.e[1][1];
    out[4].subC = {{0, -1}, {1, +1}};

    // m6: (x10 - x00)*(y00 + y01) -> Csub: 22 (+1)
    out[5].LA = X.e[1][0]; lin_accum(out[5].LA, X.e[0][0], -1);
    out[5].LB = Y.e[0][0]; lin_accum(out[5].LB, Y.e[0][1], +1);
    out[5].subC = {{3, +1}};

    // m7: (x01 - x11)*(y10 + y11) -> Csub: 11 (+1)
    out[6].LA = X.e[0][1]; lin_accum(out[6].LA, X.e[1][1], -1);
    out[6].LB = Y.e[1][0]; lin_accum(out[6].LB, Y.e[1][1], +1);
    out[6].subC = {{0, +1}};
}

// Top-level recombination mapping for each Mk to top-level C quadrants (q in {0..3}):
// q index: 0->C11, 1->C12, 2->C21, 3->C22
static inline const std::vector<std::pair<int,int>>& topC_for_k(int k){
    // Based on:
    // C11 = +M1 +M4 -M5 +M7
    // C12 =      +M3 +M5
    // C21 = +M2 +M4
    // C22 = +M1 -M2 +M3 +M6
    static const std::vector<std::pair<int,int>> tbl[7] = {
        /*k=0 (M1)*/ {{0,+1},{3,+1}},
        /*k=1 (M2)*/ {{2,+1},{3,-1}},
        /*k=2 (M3)*/ {{1,+1},{3,+1}},
        /*k=3 (M4)*/ {{0,+1},{2,+1}},
        /*k=4 (M5)*/ {{0,-1},{1,+1}},
        /*k=5 (M6)*/ {{3,+1}},
        /*k=6 (M7)*/ {{0,+1}},
    };
    return tbl[k];
}

// Top-level Strassen operands (over quadrants):
// Mk left over A-quads; right over B-quads (indices 0..3) with signs.
static inline void top_op_for_k(int k,
                                std::vector<std::pair<int,int>>& Aparts,
                                std::vector<std::pair<int,int>>& Bparts)
{
    Aparts.clear(); Bparts.clear();
    switch(k){
        case 0: // M1: (A11 + A22)*(B11 + B22)
            Aparts = {{0,+1},{3,+1}};
            Bparts = {{0,+1},{3,+1}};
            break;
        case 1: // M2: (A21 + A22)*B11
            Aparts = {{2,+1},{3,+1}};
            Bparts = {{0,+1}};
            break;
        case 2: // M3: A11*(B12 - B22)
            Aparts = {{0,+1}};
            Bparts = {{1,+1},{3,-1}};
            break;
        case 3: // M4: A22*(B21 - B11)
            Aparts = {{3,+1}};
            Bparts = {{2,+1},{0,-1}};
            break;
        case 4: // M5: (A11 + A12)*B22
            Aparts = {{0,+1},{1,+1}};
            Bparts = {{3,+1}};
            break;
        case 5: // M6: (A21 - A11)*(B11 + B12)
            Aparts = {{2,+1},{0,-1}};
            Bparts = {{0,+1},{1,+1}};
            break;
        case 6: // M7: (A12 - A22)*(B21 + B22)
            Aparts = {{1,+1},{3,-1}};
            Bparts = {{2,+1},{3,+1}};
            break;
    }
}

// Map (top-level quadrant q, sub-quadrant s) to global C-block id [0..15]
static inline int Cid_from_qs(int q, int s){
    int qr = (q/2), qc = (q%2);     // q: 0..3 -> (row of quadrants, col)
    int sr = (s/2), sc = (s%2);     // s: 0..3 -> (row inside quad, col)
    int r = qr*2 + sr;
    int c = qc*2 + sc;
    return r*4 + c;
}

// Build all 49 leaves
static std::array<Leaf,49> build_depth2_recipes(){
    std::array<Leaf,49> Ls;

    // Build base quadrants (2x2 of base Lin) for A and B
    std::array<Lin2x2,4> QA, QB;
    build_quadrants_A(QA);
    build_quadrants_B(QB);

    int idx = 0;
    for (int k=0;k<7;k++){
        // Top-level operands
        std::vector<std::pair<int,int>> Aparts, Bparts;
        top_op_for_k(k, Aparts, Bparts);

        // Combine quadrants into 2x2 Lin2x2 X (for A) and Y (for B)
        Lin2x2 X = combine_quads(QA, Aparts);
        Lin2x2 Y = combine_quads(QB, Bparts);

        // Second level: make the 7 sub-leaves m=0..6
        std::array<SubLeaf,7> subs;
        make_sub_leaves(X, Y, subs);

        // For each sub-leaf, push a Leaf with composed C mapping
        const auto& topC = topC_for_k(k); // top-level C quadrants & signs
        for (int m=0;m<7;m++){
            Leaf leaf;

            // Collapse LA, LB Lin into Aterms and Bterms (drop zeros)
            for (auto& kv: subs[m].LA){
                if (kv.second!=0) leaf.Aterms.push_back({ kv.first, (int8_t)(kv.second>=0?+1:-1) });
            }
            for (auto& kv: subs[m].LB){
                if (kv.second!=0) leaf.Bterms.push_back({ kv.first, (int8_t)(kv.second>=0?+1:-1) });
            }

            // Compose C routes: for each top quadrant q with sign tq,
            // for each sub-quadrant s with sign ts, add to final C id.
            for (auto [q, tq]: topC){
                for (auto [s, ts]: subs[m].subC){
                    int cid = Cid_from_qs(q, s);
                    int sign = tq * ts;
                    leaf.Couts.push_back({ cid, (int8_t)(sign>=0?+1:-1) });
                }
            }

            // Optional: dedupe identical Aterms/Bterms ids by summing signs (kept simple here)
            Ls[idx++] = std::move(leaf);
        }
    }
    // idx must be 49
    return Ls;
}

// ---------- Main ----------
int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int me=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (world != 49 && me==0){
        std::fprintf(stderr, "WARNING: intended to run with 49 ranks (got %d).\n", world);
    }

    // Metrics
    double t0_all = MPI_Wtime();
    double t_fetch = 0.0, t_compute = 0.0;
    unsigned long long g_bytes_recv_payload = 0ULL;

    // -------- Placement: two pins per rank ----------
    // Default placement: each rank pins one A and one B (round-robin). Replace if you have a custom layout.
    std::vector<std::pair<std::string,std::string>> placement(49);
    for (int r=0; r<49; ++r){
        int aidx = r % 16, bidx = r % 16;
        int ar = aidx/4 + 1, ac = aidx%4 + 1;
        int br = bidx/4 + 1, bc = bidx%4 + 1;
        std::ostringstream sa, sb;
        sa << "A:a" << ar << ac;
        sb << "B:b" << br << bc;
        placement[r] = {sa.str(), sb.str()};
    }
    build_owner_from_placement(placement, world);

    // -------- Local store (only blocks owned by this rank) ----------
    Store store;
    for (int id=0; id<32; ++id)
        if (owner_of(id, world) == me)
            store.data[id] = make_block(id+1);

    // -------- Build depth-2 Strassen leaves --------
    auto recipes = build_depth2_recipes();

    // -------- Execute leaf (only ranks 0..48 produce a leaf) --------
    std::array<double,64> sendbuf{}; // 16 blocks * 4 doubles
    std::array<double,64> recvbuf{};
    sendbuf.fill(0.0);
    recvbuf.fill(0.0);

    if (me < 49){
        const Leaf& leaf = recipes[me];

        // Collect unique needed A/B ids
        std::vector<int> need_ids;
        need_ids.reserve(leaf.Aterms.size()+leaf.Bterms.size());
        for (auto t: leaf.Aterms) need_ids.push_back(t.id);
        for (auto t: leaf.Bterms) need_ids.push_back(t.id);
        std::sort(need_ids.begin(), need_ids.end());
        need_ids.erase(std::unique(need_ids.begin(), need_ids.end()), need_ids.end());

        // Fetch those we don't own (typically 4–6 after the two pins)
        std::vector<int> to_fetch; to_fetch.reserve(need_ids.size());
        for (int id: need_ids) if (!store.data.count(id)) to_fetch.push_back(id);

        std::unordered_map<int,M2> cache; cache.reserve(to_fetch.size()+4);

        double t0 = MPI_Wtime();
        fetch_many_progress(to_fetch, me, world, store,
            [&](int id, const M2& blk){ cache[id] = blk; },
            g_bytes_recv_payload);
        double t1 = MPI_Wtime(); t_fetch += (t1 - t0);

        auto getblk = [&](int id)->const M2& {
            if (auto it=store.data.find(id); it!=store.data.end()) return it->second;
            return cache.at(id);
        };

        // Evaluate linear forms and compute the single 2x2 leaf product
        auto combine = [&](const std::vector<Term>& ts)->M2{
            M2 acc{}; // zero
            for (auto t: ts){
                const M2& b = getblk(t.id);
                if (t.s==+1) acc = add2(acc, b, 1.0, +1.0);
                else         acc = add2(acc, b, 1.0, -1.0);
            }
            return acc;
        };

        t0 = MPI_Wtime();
        M2 Atil = combine(leaf.Aterms);
        M2 Btil = combine(leaf.Bterms);
        M2 P    = mul2(Atil, Btil);
        double t2 = MPI_Wtime(); t_compute += (t2 - t0);

        // Build contributions to C (only the few blocks this leaf touches)
        std::array<M2,16> Ccontrib{};
        for (auto t: leaf.Couts){
            if (t.s==+1) Ccontrib[t.id] = add2(Ccontrib[t.id], P, 1.0, +1.0);
            else         Ccontrib[t.id] = add2(Ccontrib[t.id], P, 1.0, -1.0);
        }

        for (int k=0; k<16; ++k) pack_M2(Ccontrib[k], &sendbuf[4*k]);
    }

    // All reduce to rank 0
    MPI_Reduce(sendbuf.data(), recvbuf.data(), 64, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // -------- Rank 0: stitch & dump result + metrics --------
    double t_rank_total = MPI_Wtime() - t0_all;

    double sum_fetch=0, sum_compute=0, sum_total=0, max_total=0;
    unsigned long long total_bytes_recv=0ULL;

    MPI_Reduce(&t_fetch,   &sum_fetch,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute, &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_rank_total, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_rank_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_bytes_recv_payload, &total_bytes_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (me==0){
        std::array<M2,16> Cblocks{};
        for (int k=0;k<16;k++) Cblocks[k] = unpack_M2(&recvbuf[4*k]);
        double C[8][8]{}; stitch8(C, Cblocks);

        double avg_fetch   = sum_fetch   / world;
        double avg_compute = sum_compute / world;
        double avg_total   = sum_total   / world;

        std::ofstream fout("result.txt");
        fout.setf(std::ios::fixed); fout.precision(6);

        fout << "Result C (8x8 numeric matrix) — depth-2 Strassen (49 leaves)\n";
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                fout << C[r][c] << (c + 1 < 8 ? ' ' : '\n');
            }
        }
        fout << "\n--- Metrics ---\n";
        fout << "Average fetch time (s): " << avg_fetch << "\n";
        fout << "Average compute time (s): " << avg_compute << "\n";
        fout << "Average total time per rank (s): " << avg_total << "\n";
        fout << "Total job time (max over ranks, s): " << max_total << "\n";
        fout << "Total payload bytes RECEIVED (approx): " << total_bytes_recv
             << " (" << (total_bytes_recv/1024.0) << " KB)\n";
        fout << "World size: " << world << " (intended 49)\n";
        fout.close();

        std::printf("Result and metrics written to result.txt\n");
    }

    MPI_Finalize();
    return 0;
}
