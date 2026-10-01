#include "setup.hpp"
#include "info.hpp"
#include <cstdlib>
#include <cstdio>
#include <chrono>

// scratch benchmark harness: NS_SIZES="64,128,192" NS_STEPS=500 NS_DX=1
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void main_setup() {
	const char* es = getenv("NS_SIZES"); const char* et = getenv("NS_STEPS"); const char* ed = getenv("NS_DX");
	string sizes = es ? es : "128"; const ulong steps = et ? (ulong)atoll(et) : 500ull; const uint Dx = ed ? (uint)atoi(ed) : 1u;
	vector<uint> Ns; { size_t p=0; while(p<sizes.size()) { size_t q=sizes.find(',', p); if(q==string::npos) q=sizes.size(); Ns.push_back((uint)atoi(sizes.substr(p, q-p).c_str())); p=q+1; } }
	for(uint N : Ns) {
		const double tc = now_s();
		LBM lbm(N*Dx, N, N, Dx, 1u, 1u, 1.0f);
		const double t_construct = now_s()-tc;
		const double ti = now_s();
		lbm.run(0u); // initialize only
		const double t_init = now_s()-ti;
		lbm.run(10u); // warmup
		lbm.update_fields();
		const double t0 = now_s();
		lbm.run(steps);
		for(uint d=0u; d<lbm.get_D(); d++) lbm.lbm_domain[d]->finish_queue();
		const double s = now_s()-t0;
		const double mlups = (double)lbm.get_N()*(double)steps*1E-6/s;
		fprintf(stderr, "RESULT N=%u Dx=%u cells=%llu steps=%llu seconds=%.4f mlups=%.1f bw_gbs=%.2f construct_s=%.3f init_s=%.3f\n", N, Dx, (unsigned long long)lbm.get_N(), (unsigned long long)steps, s, mlups, mlups*1E-3*(double)bandwidth_bytes_per_cell_device(), t_construct, t_init);
	}
}
