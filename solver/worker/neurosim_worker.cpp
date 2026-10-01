// NeuroSim solver worker.
// Builds a FluidX3D simulation from a case file, runs it, writes JSON-lines telemetry, slices, frames, field
// exports and checkpoints into the run directory, and accepts live commands on stdin.
// Compiled together with the (patched) FluidX3D sources; replaces FluidX3D's setup.cpp and main.cpp.
#include "lbm.hpp"
#include "shapes.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>

#ifndef NEUROSIM_WORKER
#error "compile with -DNEUROSIM_WORKER"
#endif

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static string f2s(const double v) { char b[32]; snprintf(b, sizeof(b), "%.7g", v); return std::isfinite(v) ? string(b) : string("null"); }
static string vec3(const float3& v) { return "["+f2s(v.x)+","+f2s(v.y)+","+f2s(v.z)+"]"; }
static string quoted(const string& s) { string r = "\""; for(char c : s) { if(c=='"'||c=='\\') r += '\\'; r += c; } return r+"\""; }

// ---------------------------------------------------------------- telemetry and commands
static std::ofstream telemetry;
static std::mutex telemetry_lock;
static void emit(const string& json) {
	std::lock_guard<std::mutex> lock(telemetry_lock);
	telemetry << json << '\n';
	telemetry.flush();
}

static std::deque<string> commands;
static std::mutex commands_lock;
static std::atomic_bool stdin_closed = false;
static void read_stdin() {
	string line;
	while(std::getline(std::cin, line)) {
		std::lock_guard<std::mutex> lock(commands_lock);
		commands.push_back(line);
	}
	stdin_closed = true; // parent went away: stop cleanly
}
static bool next_command(string& line) {
	std::lock_guard<std::mutex> lock(commands_lock);
	if(commands.empty()) return false;
	line = commands.front();
	commands.pop_front();
	return true;
}

// ---------------------------------------------------------------- NeuroSim device kernels (own program, FluidX3D's context and queue)
static const string ns_kernels = R"CLC(
#define TYPE_S 0x01
#define TYPE_G 0x20
inline bool is_fluid(const uchar f) { return !(f&(TYPE_S|TYPE_G)); }
// per-work-item partial sums: count, rho, |u|^2, max|u|, ux, uy, uz, extra (T or phi)
kernel void ns_stats(const global float* rho, const global float* u, const global uchar* flags, const global float* extra, const uint has_extra, const ulong N, global float* partial) {
	const ulong id = get_global_id(0), stride = get_global_size(0);
	float c=0.0f, sr=0.0f, su2=0.0f, mu=0.0f, sx=0.0f, sy=0.0f, sz=0.0f, se=0.0f;
	for(ulong n=id; n<N; n+=stride) {
		if(!is_fluid(flags[n])) continue;
		const float ux=u[n], uy=u[N+n], uz=u[2ul*N+n], u2=ux*ux+uy*uy+uz*uz;
		c += 1.0f; sr += rho[n]; su2 += u2; mu = fmax(mu, u2); sx += ux; sy += uy; sz += uz;
		if(has_extra) se += extra[n];
	}
	global float* p = partial+8ul*id;
	p[0]=c; p[1]=sr; p[2]=su2; p[3]=sqrt(mu); p[4]=sx; p[5]=sy; p[6]=sz; p[7]=se;
}
// 2D slice through the field; solid cells are written as NaN
kernel void ns_slice(const global float* rho, const global float* u, const global uchar* flags, const global float* extra, const uint Nx, const uint Ny, const uint Nz, const uint axis, const uint pos, const uint field, const uint step, const uint W, const uint H, global float* out) {
	const uint id = get_global_id(0);
	if(id>=W*H) return;
	const uint i = (id%W)*step, j = (id/W)*step;
	uint x, y, z;
	if(axis==0u) { x=pos; y=i; z=j; } else if(axis==1u) { x=i; y=pos; z=j; } else { x=i; y=j; z=pos; }
	x = min(x, Nx-1u); y = min(y, Ny-1u); z = min(z, Nz-1u);
	const ulong N = (ulong)Nx*Ny*Nz, n = (ulong)x+((ulong)y+(ulong)z*Ny)*Nx;
	const uchar f = flags[n];
	if(field!=7u && (f&TYPE_S)) { out[id] = NAN; return; }
	float v = 0.0f;
	const float ux=u[n], uy=u[N+n], uz=u[2ul*N+n];
	switch(field) {
		case 0u: v = sqrt(ux*ux+uy*uy+uz*uz); break;
		case 1u: v = ux; break;
		case 2u: v = uy; break;
		case 3u: v = uz; break;
		case 4u: v = rho[n]; break;
		case 5u: { // vorticity magnitude, central differences, clamped at the box edges
			const ulong xp=(ulong)min(x+1u,Nx-1u)+((ulong)y+(ulong)z*Ny)*Nx, xm=(ulong)(x>0u?x-1u:0u)+((ulong)y+(ulong)z*Ny)*Nx;
			const ulong yp=(ulong)x+((ulong)min(y+1u,Ny-1u)+(ulong)z*Ny)*Nx, ym=(ulong)x+((ulong)(y>0u?y-1u:0u)+(ulong)z*Ny)*Nx;
			const ulong zp=(ulong)x+((ulong)y+(ulong)min(z+1u,Nz-1u)*Ny)*Nx, zm=(ulong)x+((ulong)y+(ulong)(z>0u?z-1u:0u)*Ny)*Nx;
			const float wx = 0.5f*((u[2ul*N+yp]-u[2ul*N+ym])-(u[N+zp]-u[N+zm]));
			const float wy = 0.5f*((u[zp]-u[zm])-(u[2ul*N+xp]-u[2ul*N+xm]));
			const float wz = 0.5f*((u[N+xp]-u[N+xm])-(u[yp]-u[ym]));
			v = sqrt(wx*wx+wy*wy+wz*wz);
		} break;
		case 6u: v = extra[n]; break; // temperature or fill level
		case 7u: v = (float)f; break;
		case 8u: v = (rho[n]-1.0f)/3.0f; break; // lattice pressure deviation
	}
	out[id] = v;
}
)CLC";

struct NsKernels {
	cl::CommandQueue queue;
	cl::Kernel stats, slice;
	cl::Buffer partial, slice_out;
	uint stats_items = 16384u;
	size_t slice_capacity = 0u;
	cl::Context context;
	void init(const Device& device) {
		context = device.get_cl_context();
		queue = device.get_cl_queue(); // same in-order queue as the solver, so no extra synchronization is needed
		cl::Program program(context, cl::Program::Sources{ { ns_kernels.c_str(), ns_kernels.length() } });
		if(program.build({ device.info.cl_device }, "-cl-std=CL1.2")) print_error("NeuroSim kernels failed to compile: "+program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device.info.cl_device));
		stats = cl::Kernel(program, "ns_stats");
		slice = cl::Kernel(program, "ns_slice");
		partial = cl::Buffer(context, CL_MEM_READ_WRITE, (size_t)stats_items*8u*sizeof(float));
	}
};

// ---------------------------------------------------------------- case description
struct Object {
	string kind, role, file;
	float3 p, n, l, center, omega;
	float r = 0.0f, T = 1.0f;
	Mesh* mesh = nullptr;
	uchar flag = TYPE_S;
};

int main(int argc, char* argv[]) {
	if(argc>=2 && string(argv[1])=="--devices") { // device inventory for the platform, as JSON
		const vector<Device_Info> devices = get_devices(false);
		string s = "[";
		for(uint i=0u; i<(uint)devices.size(); i++) {
			const Device_Info& d = devices[i];
			s += string(i ? "," : "")+"{\"id\":"+to_string(d.id)+",\"name\":"+quoted(d.name)+",\"vendor\":"+quoted(d.vendor)+",\"driver\":"+quoted(d.driver_version)
				+",\"opencl_c\":"+quoted(d.opencl_c_version)+",\"memory_mb\":"+to_string(d.memory)+",\"max_buffer_mb\":"+to_string(d.max_global_buffer)
				+",\"compute_units\":"+to_string(d.compute_units)+",\"clock_mhz\":"+to_string(d.clock_frequency)+",\"tflops\":"+f2s(d.tflops)
				+",\"is_gpu\":"+(d.is_gpu?"true":"false")+",\"uses_ram\":"+(d.uses_ram?"true":"false")+",\"fp16\":"+(d.is_fp16_capable?"true":"false")+",\"fp64\":"+(d.is_fp64_capable?"true":"false")+"}";
		}
		printf("%s]\n", s.c_str());
		return 0;
	}
	if(argc>=2 && string(argv[1])=="--probe") { // attainable device memory bandwidth (copy), for roofline-style analysis
		const vector<Device_Info> devices = get_devices(false);
		const Device_Info& d = devices[min((uint)(argc>=3 ? atoi(argv[2]) : 0), (uint)devices.size()-1u)];
		const size_t bytes = (size_t)min(256u, d.max_global_buffer/4u)<<20;
		cl::CommandQueue q(d.cl_context, d.cl_device);
		const string src = "kernel void copy4(global const float4* a, global float4* b) { const size_t i=get_global_id(0); b[i]=a[i]; }";
		cl::Program program(d.cl_context, cl::Program::Sources{ { src.c_str(), src.length() } });
		program.build({ d.cl_device });
		cl::Kernel k(program, "copy4");
		cl::Buffer a(d.cl_context, CL_MEM_READ_WRITE, bytes), b(d.cl_context, CL_MEM_READ_WRITE, bytes);
		const float zero = 0.0f;
		q.enqueueFillBuffer(a, zero, 0u, bytes); q.enqueueFillBuffer(b, zero, 0u, bytes);
		k.setArg(0, a); k.setArg(1, b);
		double best = 1E30;
		for(uint r=0u; r<8u; r++) { const double t0 = now_s(); q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(bytes/16u), cl::NDRange(64u)); q.finish(); if(r>0u) best = fmin(best, now_s()-t0); }
		printf("{\"device\":%s,\"copy_gbs\":%s}\n", quoted(d.name).c_str(), f2s(2.0*(double)bytes/best*1E-9).c_str());
		return 0;
	}
	if(argc<3) { fprintf(stderr, "usage: neurosim-fx <case.cfg> <run_dir> | --devices | --probe [device]\n"); return 2; }
	const string run_dir = string(argv[2])+"/";
	Configuration_File cfg(argv[1]);
	telemetry.open(run_dir+"telemetry.jsonl", std::ios::out|std::ios::app);

	const uint Nx=cfg.value<uint>("Nx"), Ny=cfg.value<uint>("Ny"), Nz=cfg.value<uint>("Nz");
	const int device_id = cfg.value<int>("device", -1);
	main_arguments = device_id>=0 ? vector<string>{ to_string(device_id) } : vector<string>(); // FluidX3D reads device IDs from here
	const string pad = to_string(cfg.value<uint>("ddf_pad", 0u));
	static string pad_env; pad_env = "NS_PAD="+pad; // putenv keeps the pointer
	set_environment_variable((char*)pad_env.c_str()); // padded DDF stride (patch 0002)
	camera = Camera(GRAPHICS_FRAME_WIDTH, GRAPHICS_FRAME_HEIGHT, 60u);

	const float3 f = float3(cfg.value<float>("fx", 0.0f), cfg.value<float>("fy", 0.0f), cfg.value<float>("fz", 0.0f));
	const float sigma=cfg.value<float>("sigma", 0.0f), alpha=cfg.value<float>("alpha", 0.0f), beta=cfg.value<float>("beta", 0.0f);
	const uint particles_N = cfg.value<uint>("particles", 0u);
	LBM lbm(Nx, Ny, Nz, 1u, 1u, 1u, cfg.value<float>("nu"), f.x, f.y, f.z, sigma, alpha, beta, particles_N, cfg.value<float>("particles_rho", 1.0f));

	// ---- initial and boundary conditions
	const vector<float> zero3 = { 0.0f, 0.0f, 0.0f };
	const vector<float> u0v = cfg.value<vector<float>>("init_u", zero3), uiv = cfg.value<vector<float>>("inflow_u", zero3), uwv = cfg.value<vector<float>>("wall_u", zero3);
	const float3 u0(u0v[0], u0v[1], u0v[2]), ui(uiv[0], uiv[1], uiv[2]), uw(uwv[0], uwv[1], uwv[2]);
	const string faces[6] = { cfg.value<string>("face_x0", "periodic"), cfg.value<string>("face_x1", "periodic"), cfg.value<string>("face_y0", "periodic"), cfg.value<string>("face_y1", "periodic"), cfg.value<string>("face_z0", "periodic"), cfg.value<string>("face_z1", "periodic") };
	const string init_mode = cfg.value<string>("init_mode", "uniform");
	const float noise = cfg.value<float>("init_noise", 0.0f), T_hot = cfg.value<float>("T_hot", 1.5f), T_cold = cfg.value<float>("T_cold", 0.5f);
	const bool hydrostatic = cfg.value<bool>("init_hydrostatic", false);
	const bool track_forces = cfg.value<bool>("track_forces", false);
	vector<Object> objects(cfg.value<uint>("obj_count", 0u));
	for(uint k=0u; k<(uint)objects.size(); k++) {
		const string p = "obj"+to_string(k)+"_";
		Object& o = objects[k];
		o.kind = cfg.value<string>(p+"kind"); o.role = cfg.value<string>(p+"role", "solid"); o.file = cfg.value<string>(p+"file", "");
		const vector<float> pv = cfg.value<vector<float>>(p+"p", zero3), nv = cfg.value<vector<float>>(p+"n", zero3), lv = cfg.value<vector<float>>(p+"l", zero3), cv = cfg.value<vector<float>>(p+"center", zero3), wv = cfg.value<vector<float>>(p+"omega", zero3);
		o.p = float3(pv[0], pv[1], pv[2]); o.n = float3(nv[0], nv[1], nv[2]); o.l = float3(lv[0], lv[1], lv[2]); o.center = float3(cv[0], cv[1], cv[2]); o.omega = float3(wv[0], wv[1], wv[2]);
		o.r = cfg.value<float>(p+"r", 0.0f); o.T = cfg.value<float>(p+"T", 1.0f);
		o.flag = o.role=="fluid" ? TYPE_F : o.role=="heat" ? TYPE_T : (uchar)(TYPE_S|TYPE_X); // TYPE_X marks object cells (forces, zero initial velocity)
	}
	const uint fill_count = cfg.value<uint>("fill_count", 0u);
	vector<vector<float>> fills(fill_count);
	for(uint k=0u; k<fill_count; k++) fills[k] = cfg.value<vector<float>>("fill"+to_string(k), vector<float>{ 0, 0, 0, 0, 0, 0 });

	uint seed0 = cfg.value<uint>("seed", 42u);
	const uint threads = (uint)thread::hardware_concurrency();
	vector<uint> seeds(threads); for(uint t=0u; t<threads; t++) seeds[t] = seed0+t;
	const float A = cfg.value<float>("tg_amplitude", 0.05f);
	parallel_for(lbm.get_N(), threads, [&](ulong n, uint t) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		float3 u = u0;
		if(init_mode=="taylor_green") {
			const float fx_=2.0f*pif*((float)x+0.5f)/(float)Nx, fy_=2.0f*pif*((float)y+0.5f)/(float)Ny, fz_=2.0f*pif*((float)z+0.5f)/(float)Nz;
			u = float3(A*cosf(fx_)*sinf(fy_)*sinf(fz_), -A*sinf(fx_)*cosf(fy_)*sinf(fz_), 0.0f);
			lbm.rho[n] = 1.0f-sq(A)*3.0f/16.0f*(cosf(2.0f*fx_)+cosf(2.0f*fy_))*(cosf(2.0f*fz_)+2.0f);
		}
		if(noise>0.0f) u += float3(random_symmetric(seeds[t], noise), random_symmetric(seeds[t], noise), random_symmetric(seeds[t], noise));
		lbm.u.x[n] = u.x; lbm.u.y[n] = u.y; lbm.u.z[n] = u.z;
		if(hydrostatic) lbm.rho[n] = units.rho_hydrostatic(-f.z, (float)z, 0.5f*(float)Nz);
		for(const vector<float>& b : fills) if((float)x>=b[0]&&(float)y>=b[1]&&(float)z>=b[2]&&(float)x<b[3]&&(float)y<b[4]&&(float)z<b[5]) lbm.flags[n] = TYPE_F;
		for(const Object& o : objects) {
			bool inside = false;
			if(o.kind=="sphere") inside = sphere(x, y, z, o.p, o.r);
			else if(o.kind=="cylinder") inside = cylinder(x, y, z, o.p, o.n, o.r);
			else if(o.kind=="cuboid") inside = cuboid(x, y, z, o.p, o.l);
			if(!inside) continue;
			lbm.flags[n] = o.flag;
			if(o.flag&TYPE_S) lbm.u.x[n] = lbm.u.y[n] = lbm.u.z[n] = 0.0f; // a solid with velocity would act as a moving wall
#ifdef TEMPERATURE
			if(o.role=="heat") lbm.T[n] = o.T;
#endif // TEMPERATURE
		}
		const uint c[6] = { x, Nx-1u-x, y, Ny-1u-y, z, Nz-1u-z };
		for(uint i=0u; i<6u; i++) { // later faces override earlier ones at edges
			const string& v = faces[i];
			if(c[i]==0u) {
				if(v=="wall"||v=="hot"||v=="cold") { lbm.flags[n] = TYPE_S; lbm.u.x[n]=lbm.u.y[n]=lbm.u.z[n]=0.0f; }
				else if(v=="moving") { lbm.flags[n] = TYPE_S; lbm.u.x[n]=uw.x; lbm.u.y[n]=uw.y; lbm.u.z[n]=uw.z; }
				else if(v=="freestream") { lbm.flags[n] = TYPE_E; lbm.u.x[n]=ui.x; lbm.u.y[n]=ui.y; lbm.u.z[n]=ui.z; lbm.rho[n] = 1.0f; }
#ifdef TEMPERATURE
			} else if(c[i]==1u && (v=="hot"||v=="cold") && lbm.flags[n]!=TYPE_S) {
				lbm.flags[n] = TYPE_T; lbm.T[n] = v=="hot" ? T_hot : T_cold;
#endif // TEMPERATURE
			}
		}
	});
#ifdef PARTICLES
	{ uint s = seed0; for(ulong i=0ull; i<lbm.particles->length(); i++) {
		lbm.particles->x[i] = random_symmetric(s, 0.5f*(float)Nx-1.0f); lbm.particles->y[i] = random_symmetric(s, 0.5f*(float)Ny-1.0f); lbm.particles->z[i] = random_symmetric(s, 0.5f*(float)Nz-1.0f);
	} }
#endif // PARTICLES
	vector<Object*> rotating;
	lbm.flags.write_to_device(); lbm.u.write_to_device(); // device voxelization below reads back flags/u, so upload the host-side setup first
	for(Object& o : objects) {
		if(o.kind!="stl") continue;
		o.mesh = read_stl(o.file); // already in lattice coordinates (prepared by the platform)
		o.mesh->set_center(o.center);
		if(length(o.omega)>0.0f) rotating.push_back(&o);
		else lbm.voxelize_mesh_on_device(o.mesh, o.flag);
	}
	parallel_for(lbm.get_N(), [&](ulong n) { if((lbm.flags[n]&(TYPE_S|TYPE_X))==(TYPE_S|TYPE_X)) lbm.u.x[n] = lbm.u.y[n] = lbm.u.z[n] = 0.0f; }); // voxelized static objects: no initial wall velocity
	const uint rot_dt = max(1u, cfg.value<uint>("rotation_dt", 4u));

	// ---- graphics defaults
	lbm.graphics.visualization_modes = cfg.value<int>("vis_modes", VIS_FLAG_SURFACE|VIS_Q_CRITERION);
	lbm.graphics.field_mode = cfg.value<int>("vis_field", 0);
	lbm.graphics.slice_mode = cfg.value<int>("vis_slice", 0);
	{ const vector<float> cam = cfg.value<vector<float>>("camera", vector<float>{ 30.0f, 20.0f, 60.0f, 1.0f }); lbm.graphics.set_camera_centered(cam[0], cam[1], cam[2], cam[3]); }

	// ---- run control
	const ulong steps = cfg.value<ulong>("steps", 0ull); // 0 = until stopped
	ulong telemetry_every = max(1ull, cfg.value<ulong>("telemetry_every", 100ull));
	ulong slice_every = cfg.value<ulong>("slice_every", 200ull);
	ulong checkpoint_every = cfg.value<ulong>("checkpoint_every", 0ull);
	uint slice_axis = cfg.value<uint>("slice_axis", 1u), slice_field = cfg.value<uint>("slice_field", 0u), slice_max = cfg.value<uint>("slice_max", 384u);
	uint slice_pos = cfg.value<uint>("slice_pos", (slice_axis==0u?Nx:slice_axis==1u?Ny:Nz)/2u);
	const bool has_T = cfg.value<bool>("has_T", false), has_phi = cfg.value<bool>("has_phi", false);

	const double t_build0 = now_s();
	lbm.run(0u); // copy initial state to the device and initialize DDFs
	for(Object* o : rotating) lbm.voxelize_mesh_on_device(o->mesh, o->flag, o->center, float3(0.0f), o->omega);
	LBM_Domain& dom = *lbm.lbm_domain[0];
	const Device& device = dom.get_device();
	NsKernels ns; ns.init(device);
	const cl::Buffer& extra_buffer =
#ifdef TEMPERATURE
		has_T ? dom.T.get_cl_buffer() :
#endif // TEMPERATURE
#ifdef SURFACE
		has_phi ? dom.phi.get_cl_buffer() :
#endif // SURFACE
		dom.rho.get_cl_buffer();
	const uint has_extra = (has_T||has_phi) ? 1u : 0u;

	// ---- checkpoint / restart (raw device buffers + time step; requires identical case and build)
	auto buffer_io = [&](std::fstream& file, const cl::Buffer& b, const ulong bytes, const bool write_to_file) {
		vector<char> host(bytes);
		if(write_to_file) { ns.queue.enqueueReadBuffer(b, CL_TRUE, 0u, bytes, host.data()); file.write(host.data(), bytes); }
		else { file.read(host.data(), bytes); ns.queue.enqueueWriteBuffer(b, CL_TRUE, 0u, bytes, host.data()); }
	};
	auto state_io = [&](const string& path, const bool save) {
		if(save) lbm.update_fields();
		std::fstream file(path, (save ? std::ios::out : std::ios::in)|std::ios::binary);
		if(!file) { emit("{\"type\":\"error\",\"message\":"+quoted("cannot open checkpoint "+path)+"}"); return false; }
		ulong header[6] = { 0x4B43534Eull, Nx, Ny, Nz, lbm.get_t(), sizeof(fpxx)|((ulong)atoll(pad.c_str())<<8) };
		if(save) file.write((char*)header, sizeof(header));
		else {
			ulong h[6]; file.read((char*)h, sizeof(h));
			if(h[0]!=header[0]||h[1]!=Nx||h[2]!=Ny||h[3]!=Nz||h[5]!=header[5]) { emit("{\"type\":\"error\",\"message\":\"checkpoint does not match this case/build\"}"); return false; }
			dom.set_t(h[4]);
		}
		buffer_io(file, dom.get_fi().get_cl_buffer(), dom.get_fi().capacity(), save);
		buffer_io(file, dom.flags.get_cl_buffer(), dom.flags.capacity(), save);
		buffer_io(file, dom.rho.get_cl_buffer(), dom.rho.capacity(), save);
		buffer_io(file, dom.u.get_cl_buffer(), dom.u.capacity(), save);
#ifdef TEMPERATURE
		buffer_io(file, dom.get_gi().get_cl_buffer(), dom.get_gi().capacity(), save);
		buffer_io(file, dom.T.get_cl_buffer(), dom.T.capacity(), save);
#endif // TEMPERATURE
#ifdef SURFACE
		buffer_io(file, dom.phi.get_cl_buffer(), dom.phi.capacity(), save);
		buffer_io(file, dom.get_mass().get_cl_buffer(), dom.get_mass().capacity(), save);
		buffer_io(file, dom.get_massex().get_cl_buffer(), dom.get_massex().capacity(), save);
#endif // SURFACE
		if(!save) { lbm.flags.read_from_device(); } // keep host copies consistent
		return true;
	};
	const string restart = cfg.value<string>("restart", "");
	if(restart!="") { if(state_io(restart, false)) emit("{\"type\":\"restart\",\"t\":"+to_string(lbm.get_t())+",\"file\":"+quoted(restart)+"}"); }

	// ---- outputs
	auto write_npy = [&](const string& name, const float* data, const uint components) {
		const string path = run_dir+"fields/"+name+"_"+to_string(lbm.get_t())+".npy";
		create_folder(path);
		string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("+(components>1u ? to_string(components)+", " : string(""))+to_string(Nz)+", "+to_string(Ny)+", "+to_string(Nx)+"), }";
		while((10u+header.length()+1u)%64u!=0u) header += ' ';
		header += '\n';
		std::ofstream file(path, std::ios::out|std::ios::binary);
		const unsigned short hl = (unsigned short)header.length();
		file.write("\x93NUMPY\x01\x00", 8); file.write((const char*)&hl, 2); file.write(header.c_str(), hl);
		file.write((const char*)data, (std::streamsize)((ulong)components*lbm.get_N()*sizeof(float)));
		return path;
	};
	auto export_fields = [&]() {
		lbm.rho.read_from_device(); lbm.u.read_from_device();
		string files = quoted(write_npy("rho", dom.rho.data(), 1u))+","+quoted(write_npy("u", dom.u.data(), 3u));
#ifdef TEMPERATURE
		lbm.T.read_from_device(); files += ","+quoted(write_npy("T", dom.T.data(), 1u));
#endif // TEMPERATURE
#ifdef SURFACE
		lbm.phi.read_from_device(); files += ","+quoted(write_npy("phi", dom.phi.data(), 1u));
#endif // SURFACE
		emit("{\"type\":\"export\",\"t\":"+to_string(lbm.get_t())+",\"files\":["+files+"]}");
	};
	double render_s = 0.0, slice_s = 0.0, last_render_dt = 0.0; // time spent outside the solver, reported for performance analytics
	uint slice_seq = 0u;
	auto write_slice = [&]() {
		const double s0 = now_s();
		lbm.update_fields();
		const uint L[3] = { Nx, Ny, Nz };
		const uint a = slice_axis==0u ? 1u : 0u, b = slice_axis==2u ? 1u : 2u; // in-plane axes
		const uint step = max(1u, (max(L[a], L[b])+slice_max-1u)/slice_max);
		const uint W = (L[a]+step-1u)/step, H = (L[b]+step-1u)/step;
		const size_t bytes = (size_t)W*H*sizeof(float);
		if(ns.slice_capacity<bytes) { ns.slice_out = cl::Buffer(ns.context, CL_MEM_READ_WRITE, bytes); ns.slice_capacity = bytes; }
		const uint pos = min(slice_pos, L[slice_axis]-1u);
		cl::Kernel& k = ns.slice;
		k.setArg(0, dom.rho.get_cl_buffer()); k.setArg(1, dom.u.get_cl_buffer()); k.setArg(2, dom.flags.get_cl_buffer()); k.setArg(3, extra_buffer);
		k.setArg(4, Nx); k.setArg(5, Ny); k.setArg(6, Nz); k.setArg(7, slice_axis); k.setArg(8, pos); k.setArg(9, slice_field); k.setArg(10, step); k.setArg(11, W); k.setArg(12, H); k.setArg(13, ns.slice_out);
		ns.queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(((size_t)W*H+63u)/64u*64u), cl::NDRange(64u));
		vector<float> host((size_t)W*H);
		ns.queue.enqueueReadBuffer(ns.slice_out, CL_TRUE, 0u, bytes, host.data());
		float vmin = 1E30f, vmax = -1E30f;
		for(float v : host) if(std::isfinite(v)) { vmin = fmin(vmin, v); vmax = fmax(vmax, v); }
		const string name = "slice_"+to_string(slice_seq++%2u)+".f32";
		std::ofstream(run_dir+"live/"+name, std::ios::out|std::ios::binary).write((const char*)host.data(), (std::streamsize)bytes);
		emit("{\"type\":\"slice\",\"t\":"+to_string(lbm.get_t())+",\"file\":"+quoted(name)+",\"w\":"+to_string(W)+",\"h\":"+to_string(H)+",\"step\":"+to_string(step)
			+",\"axis\":"+to_string(slice_axis)+",\"pos\":"+to_string(pos)+",\"field\":"+to_string(slice_field)+",\"min\":"+f2s(vmin)+",\"max\":"+f2s(vmax)+"}");
		slice_s += now_s()-s0;
	};
	uint frame_seq = 0u;
	auto write_frame = [&]() { // raw 0x00RRGGBB pixels (= Qt RGB32), double-buffered, no encoding
		const double r0 = now_s();
		const int* bitmap = lbm.graphics.draw_frame();
		const string name = "frame_"+to_string(frame_seq%2u)+".raw";
		std::ofstream(run_dir+"live/"+name, std::ios::out|std::ios::binary).write((const char*)bitmap, (std::streamsize)camera.width*camera.height*4u);
		last_render_dt = now_s()-r0;
		render_s += last_render_dt;
		emit("{\"type\":\"frame\",\"t\":"+to_string(lbm.get_t())+",\"seq\":"+to_string(frame_seq)+",\"file\":"+quoted(name)+",\"w\":"+to_string(camera.width)+",\"h\":"+to_string(camera.height)+"}");
		frame_seq++;
	};	auto stats = [&](const double mlups, const double sps, const double wall) {
		lbm.update_fields();
		const ulong N = lbm.get_N();
		cl::Kernel& k = ns.stats;
		k.setArg(0, dom.rho.get_cl_buffer()); k.setArg(1, dom.u.get_cl_buffer()); k.setArg(2, dom.flags.get_cl_buffer()); k.setArg(3, extra_buffer); k.setArg(4, has_extra); k.setArg(5, N); k.setArg(6, ns.partial);
		ns.queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(ns.stats_items), cl::NDRange(64u));
		vector<float> p((size_t)ns.stats_items*8u);
		ns.queue.enqueueReadBuffer(ns.partial, CL_TRUE, 0u, p.size()*sizeof(float), p.data());
		double c=0.0, sr=0.0, su2=0.0, mu=0.0, sx=0.0, sy=0.0, sz=0.0, se=0.0;
		for(uint i=0u; i<ns.stats_items; i++) { const float* q = &p[8u*i]; c+=q[0]; sr+=q[1]; su2+=q[2]; mu=fmax(mu, (double)q[3]); sx+=q[4]; sy+=q[5]; sz+=q[6]; se+=q[7]; }
		const double ci = c>0.0 ? 1.0/c : 0.0;
		string s = "{\"type\":\"step\",\"t\":"+to_string(lbm.get_t())+",\"mlups\":"+f2s(mlups)+",\"steps_per_s\":"+f2s(sps)+",\"fluid_cells\":"+f2s(c)
			+",\"rho_mean\":"+f2s(sr*ci)+",\"u_max\":"+f2s(mu)+",\"u_mean\":"+vec3(float3((float)(sx*ci), (float)(sy*ci), (float)(sz*ci)))+",\"ke\":"+f2s(0.5*su2*ci);
		s += ",\"render_frac\":"+f2s(wall>0.0 ? render_s/wall : 0.0)+",\"slice_frac\":"+f2s(wall>0.0 ? slice_s/wall : 0.0);
		render_s = slice_s = 0.0;
		if(has_T) s += ",\"T_mean\":"+f2s(se*ci);
		if(has_phi) s += ",\"liquid_volume\":"+f2s(se);
#ifdef FORCE_FIELD
		if(track_forces) s += ",\"force\":"+vec3(lbm.object_force(TYPE_S|TYPE_X));
#endif // FORCE_FIELD
		emit(s+"}");
	};
	auto checkpoint = [&]() {
		const string path = run_dir+"checkpoints/t"+to_string(lbm.get_t())+".nsck";
		create_folder(path);
		if(state_io(path, true)) emit("{\"type\":\"checkpoint\",\"t\":"+to_string(lbm.get_t())+",\"file\":"+quoted(path)+"}");
	};

	create_folder(run_dir+"live/x");
	emit("{\"type\":\"start\",\"N\":["+to_string(Nx)+","+to_string(Ny)+","+to_string(Nz)+"],\"device\":"+quoted(device.info.name)+",\"memory_mb\":"+to_string(device.info.memory_used)
		+",\"bytes_per_cell\":"+to_string(bytes_per_cell_device())+",\"bandwidth_bytes_per_cell\":"+to_string(bandwidth_bytes_per_cell_device())+",\"setup_s\":"+f2s(now_s()-t_build0)+",\"t\":"+to_string(lbm.get_t())+"}");
	if(const cl_int e = ns.queue.finish()) { emit("{\"type\":\"error\",\"message\":\"OpenCL device error "+to_string(e)+" during setup\"}"); _exit(3); }
	thread(read_stdin).detach();
	write_slice(); write_frame();

	bool paused = cfg.value<bool>("start_paused", false), stop = false;
	double frame_fps = cfg.value<float>("frame_fps", 8.0f), last_frame = now_s();
	double render_budget = cfg.value<float>("render_budget", 0.2f); // max share of wall time spent on live frames
	ulong next_tel = lbm.get_t()+telemetry_every, next_slice = slice_every ? lbm.get_t()+slice_every : max_ulong, next_ckpt = checkpoint_every ? lbm.get_t()+checkpoint_every : max_ulong;
	ulong t_mark = lbm.get_t(); double clock_mark = now_s(), compute_s = 0.0, steps_per_s = 0.0;
	while(!stop) {
		string line;
		while(next_command(line)) {
			std::istringstream in(line); string cmd; in >> cmd;
			if(cmd=="pause") paused = true;
			else if(cmd=="resume") paused = false;
			else if(cmd=="stop") stop = true;
			else if(cmd=="cam") { float rx=30, ry=20, fov=60, zoom=1; in >> rx >> ry >> fov >> zoom; lbm.graphics.set_camera_centered(rx, ry, fov, zoom); write_frame(); last_frame = now_s(); }
			else if(cmd=="vis") { int m=0, fm=0, sm=0; in >> m >> fm >> sm; lbm.graphics.visualization_modes = m; lbm.graphics.field_mode = fm; lbm.graphics.slice_mode = sm; in >> lbm.graphics.slice_x >> lbm.graphics.slice_y >> lbm.graphics.slice_z; write_frame(); }
			else if(cmd=="slice") { in >> slice_axis >> slice_pos >> slice_field; slice_axis = min(slice_axis, 2u); write_slice(); }
			else if(cmd=="rates") { in >> telemetry_every >> slice_every >> frame_fps; telemetry_every = max(1ull, telemetry_every); next_tel = lbm.get_t()+telemetry_every; next_slice = slice_every ? lbm.get_t()+slice_every : max_ulong; }
			else if(cmd=="budget") { in >> render_budget; }
			else if(cmd=="frame") write_frame();
			else if(cmd=="export") export_fields();
			else if(cmd=="checkpoint") checkpoint();
		}
		if(stdin_closed) stop = true;
		if(stop) break;
		if(paused || (steps && lbm.get_t()>=steps)) {
			if(steps && lbm.get_t()>=steps && !paused) { emit("{\"type\":\"finished\",\"t\":"+to_string(lbm.get_t())+"}"); paused = true; }
			sleep(0.01); clock_mark = now_s(); t_mark = lbm.get_t(); compute_s = 0.0;
			continue;
		}
		// chunk length: bounded by the next scheduled output and by ~1/fps of wall time, so commands and frames stay responsive
		const double budget = frame_fps>0.0 ? fmin(0.1, 1.0/frame_fps) : 0.25;
		ulong chunk = min(min(next_tel, next_slice), next_ckpt)-lbm.get_t();
		chunk = min(chunk, max(1ull, (ulong)(steps_per_s*budget)));
		if(steps) chunk = min(chunk, steps-lbm.get_t());
		if(!rotating.empty()) chunk = min(chunk, (ulong)rot_dt);
		chunk = max(chunk, 1ull);
		const double c0 = now_s();
		lbm.run(chunk); // asynchronous enqueue (patch 0001)
		if(const cl_int e = ns.queue.finish()) { // device faults would otherwise only show up as silently wrong data
			emit("{\"type\":\"error\",\"message\":\"OpenCL device error "+to_string(e)+" during time stepping\"}");
			break;
		}
		const double dc = now_s()-c0;
		compute_s += dc;
		steps_per_s = (double)chunk/fmax(dc, 1E-6);
		for(Object* o : rotating) { // rotate geometry and re-voxelize with matching surface velocity
			o->mesh->rotate(float3x3(normalize(o->omega), length(o->omega)*(float)chunk));
			lbm.voxelize_mesh_on_device(o->mesh, o->flag, o->center, float3(0.0f), o->omega);
		}
		const ulong t = lbm.get_t();
		if(t>=next_tel) {
			const double wall = now_s()-clock_mark, steps_done = (double)(t-t_mark);
			stats(compute_s>0.0 ? (double)lbm.get_N()*steps_done/compute_s*1E-6 : 0.0, wall>0.0 ? steps_done/wall : 0.0, wall);
			t_mark = t; clock_mark = now_s(); compute_s = 0.0;
			next_tel = t+telemetry_every;
		}
		if(t>=next_slice) { write_slice(); next_slice = slice_every ? t+slice_every : max_ulong; }
		if(frame_fps>0.0 && now_s()-last_frame>=fmax(1.0/frame_fps, last_render_dt/fmax(render_budget, 0.01))) { write_frame(); last_frame = now_s(); }
		if(t>=next_ckpt) { checkpoint(); next_ckpt = t+checkpoint_every; }
	}	emit("{\"type\":\"stopped\",\"t\":"+to_string(lbm.get_t())+"}");
	telemetry.close();
	_exit(0); // skip static destructors of detached threads
}
