#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <signal.h>
#include <float.h>
#include "linear_algebra.h"
#include "reflective_newton.h"
#include "../FEM2D/FEM2D.h"
#include "../FEM2D/geometry2D.h"
#include "../FEM2D/FEM.h"
#include "../FEM2D/conditionfile.h"
#include "../FEM2D/K2_tree.h"
#include "umfpack.h"

//#include "dmumps_c.h"

#ifdef __CUDACC__
#include "cudaLinA.h"
#endif

#define X_SHEAR
#define CONV_CHECK
//#define TEMP_EQUILIB
#define MOBILITY_ARRHENIUS
//#define X_STRETCH
//#define ANISOTROPIC_ELASTICITY
//#define X_MARGIN
#define CONSTANT_PHASE 
//#define Y_MARGIN
//#define NO_MARGIN
#define NO_TEMP
//#define OTHER
//#define FORCE_ELEMENT_BASED_INTERPOLATION

#define NO 0
#define YES 1
#define FAILED 0
#define SUCCESS 1
#define NOCON 0
#define DIRICHLET 1
#define NEUMANN 2
#define ROBIN 3
#define MIXED 4
#define LOAD_NONE "none"
#define LAST_INDEX -1
#define PREV_LAST_INDEX -2
#define QUIET 1
#define CHATTY 0
#define REFLECTIVE_NEWTON 1
#define REFLECTIVE_NEWTON_STRING "reflective Newton"
#define ACTIVE_SET 2
#define ACTIVE_SET_STRING "primal dual active set"
#define BARYCENTRIC 1
#define ELEMENT_BASE 2
#define TRACE 1
#define EIGENSYSTEM 2
#define TRACELESS 0
#define DEVIATOR 1
#define PARTIAL0 0
#define PARTIAL1 1
#define PARTIAL2 2
#define FACTOR -1
#define NO_REJECTION 0
#define PARTIAL_REJECTION 1
#define TOTAL_REJECTION 2
#define FREE_EXPAND 0
#define NO_EXPAND 1

// parameter names //////////////////////////////

#define P_COUNTER 71

// data i/o
#define P_OUTPUTNAME "output name"
#define P_OUTPUTDIR "output directory"
#define P_MESHNAME "mesh name"
#define P_MESHDIR "mesh directory"
#define P_SAMPLE_INTERVAL "sample interval"
#define P_GLOBNUM "number of global output variables"
#define P_LOAD "load from file"
#define P_COND "condition file"

// model parameters
#define P_MU1 "Lame parameter mu phase 1"
#define P_MU2 "Lame parameter mu phase 2"
#define P_LAMBDA1 "Lame parameter lambda phase 1"
#define P_LAMBDA2 "Lame parameter lambda phase 2"
#define P_ALPHA1 "rate independent damage threshold phase 1"
#define P_ALPHA2 "rate independent damage threshold phase 2"
#define P_BETA "rate dependent damage rate beta"
#define P_KAPPA1 "damage interface energy kappa phase 1"
#define P_KAPPA2 "damage interface energy kappa phase 2"
#define P_XI "damage self potential xi"
#define P_ETA_EPS "elastic regularization eta_eps"
#define P_MOBILITY1 "mobility coefficient phase 1"
#define P_MOBILITY2 "mobility coefficient phase 2"
#define P_KAPPA_C "phase interface energy kappa_c"
#define P_LANDAU_POT "Landau potential factor pot_factor"
#define P_VISCOSITY_C "phase viscosity vis_c"
#define P_T_CRITICAL "critical temperature for phase separation"
#define P_T_INITIAL "initial temperature"
#define P_EUTECTIC "eutectic point"
#define P_THERMAL_EXPANSION1 "thermal expansion coefficient phase 1"
#define P_THERMAL_EXPANSION2 "thermal expansion coefficient phase 2"
#define P_HEAT_CAPACITY0 "heat capacity damaged"
#define P_HEAT_CAPACITY1 "heat capacity phase 1"
#define P_HEAT_CAPACITY2 "heat capacity phase 2"
#define P_THERMAL_CONDUCT0 "thermal conductivity damaged"
#define P_THERMAL_CONDUCT1 "thermal conductivity phase 1"
#define P_THERMAL_CONDUCT2 "thermal conductivity phase 2"
#define P_HEAT_FLUX_OUT "boundary heat flux rate"
#define P_T_BATH "temperature of outer heat bath"
#define P_MARGIN_SIZE "margin size"
#define P_MARGIN_TOUGHNESS "toughness at margin"
#define P_ROBIN_STIFFNESS "stiffness at Robin bounds"

// backtracking
#define P_BACKTRACKING "backtracking activated"
#define P_ETOL "backtracking energy tolerance"

// time discretization
#define P_TOTALTIME "total time T"
#define P_TIMESTEP "time step dt"
#define P_TIMESTEP_NUM "number of timesteps"
#define P_TIME_ADAPTIVE "time adaptive"
#define P_TIME_ADAPTIVE_DZ "dz threshold for adaptive time step"
#define P_TIME_MIN_DT "minimal timestep as fraction of dt_max"
#define P_CONSTRAINT_THRES "threshold for constraint work"

// space discretization
#define P_SPACE_ADAPTIVE "space adaptive"
#define P_MIN_REFINEMENT "minimum refinement area fraction"
#define P_MAX_REFINEMENT "maximum refinement area fraction"
#define P_MIN_EDGE_LEN "minimum edge length"
#define P_EDGE_FACTOR "edge refinement factor"
#define P_INTERPOLATION_TOLERANCE "interpolation tolerance"
#define P_STRESS_FRACTION "crit stress refinement fraction of alpha"
#define P_STRESS_MAX_REFINE_AREA "max refined area by stress refinement"

// numerics
#define P_CONSTRAINT_SOLVER "constraint solver"

	// alternate minimization
#define P_ALT_EPS "alternate minimization eps"
#define P_ALT_MAXITER "alternate minimization max iterations"
#define P_ALT_INFO "alternate minimization print info"
#define P_ALT_TRUST_MIN "alternate minimization min trust radius"
#define P_ALT_TRUST_MAX "alternate minimization max trust radius"
#define P_ALT_TRUST_L "alternate minimization Newton sigma_l"
#define P_ALT_TRUST_U "alternate minimization Newton sigma_u"

	// reflective Newton

#define P_R_NEWTON_EPS "reflective Newton eps"
#define P_R_NEWTON_MAXITER "reflective Newton max iterations"
#define P_R_NEWTON_SIGMA_L "reflective Newton sigma_l"
#define P_R_NEWTON_SIGMA_U "reflective Newton sigma_u"
#define P_R_NEWTON_RHO "reflective Newton rho"
#define P_R_NEWTON_DELTA "reflective Newton initial delta"
#define P_R_NEWTON_TRUST_L "reflective Newton trust lower"
#define P_R_NEWTON_TRUST_U "reflective Newton trust upper"
#define P_R_NEWTON_TRUST_MIN "reflective Newton min trust radius"
#define P_R_NEWTON_TRUST_MAX "reflective Newton max trust radius"
#define P_R_NEWTON_SUB_TOL "reflective Newton subspace solver eps"
#define P_R_NEWTON_INFO "reflective Newton print info"
	
	// BiCGStab
#define P_BI_EPS "BiCGStab eps"
#define P_BI_MAXITER "BiCGStab max iterations"
#define P_BI_INFO "BiCGStab print info"
	
	// AMG
#define P_AMG_EPS "AMG eps"
#define P_AMG_MAXITER "AMG max V-cycle iterations"
#define P_AMG_SMOOTH_ITER "AMG smoothing iterations"
#define P_AMG_DEPTH "AMG depth"
#define P_AMG_MATRIX_TOL "AMG tolerance for matrix element removal"
#define P_AMG_info "AMG print info"

// Short names

#define PS_LOAD "load"

#define PS_OUTPUTNAME "out"
#define PS_OUTPUTDIR "dir"
#define PS_MESHNAME "mesh"
#define PS_COND "cond"
#define PS_SAMPLE_INTERVAL "interval"

#define PS_MU1 "mu1"
#define PS_LAMBDA1 "la1"
#define PS_MU2 "mu2"
#define PS_LAMBDA2 "la2"
#define PS_MODULI1 "mod1"
#define PS_MODULI2 "mod2"
#define PS_ALPHA1 "alpha1"
#define PS_ALPHA2 "alpha2"
#define PS_BETA "beta"
#define PS_KAPPA1 "kappa1"
#define PS_KAPPA2 "kappa2"
#define PS_XI "xi"
#define PS_ETA_EPS "eta_eps"
#define PS_MOBILITY1 "mobility1"
#define PS_MOBILITY2 "mobility2"
#define PS_KAPPA_C "kappa_c"
#define PS_EUTECTIC "eutectic"
#define PS_ROBIN_STIFFNESS "robin"
#define PS_THERMAL_CONDUCT0 "kappa_th0"
#define PS_THERMAL_CONDUCT1 "kappa_th1"
#define PS_THERMAL_CONDUCT2 "kappa_th2"
#define PS_THERMAL_EXPANSION1 "exp1"
#define PS_THERMAL_EXPANSION2 "exp2"

#define PS_TOTALTIME "time"
#define PS_TIMESTEP "dt"
#define PS_TIMESTEP_NUM "timesteps"
#define PS_TIME_ADAPTIVE "time_adaptive"
#define PS_CONSTRAINT_THRES "constraint_thres"
#define PS_REFINE_AMIN "a_min"
#define PS_TIME_MIN_DT "dt_min"
#define PS_ELEMENT_INTERPOLATION "element_interpol"

#define PS_SPACE_ADAPTIVE "space_adaptive"

#define PS_BACKTRACKING "backtracking"

#define PS_ALT_INFO "alt_info"

#define PS_TIME_OFFSET "offset_t"

#define LOAD_INDEX "index"

#define OMP_PROC_NUM "procs"


#define ES_INT_ALPHA "alpha_int"
#define ES_INT_COEFF "alpha_coeff"


// field names

#define F_PRIM 6
#define F_SHIFT_X "Displacement field x"
#define F_SHIFT_Y "Displacement field y"
#define F_DAMAGE "Damage field"
#define F_PHASE "Material phase field"
#define F_MU "Chemical potential field"
#define F_TEMP "Temperature field"

#define F_SEC 6
#define F_ADD "additional field"
//#define F_ADD2 "additional field 2"

#define F_GLOB_VAR 18

#define FIELD_NAMES {F_SHIFT_X,F_SHIFT_Y,F_DAMAGE,F_PHASE,F_MU,F_TEMP}

// extern variables (from FEM2D.c)

extern mesh2D glob_mesh;
extern element_collection elements;

// types //////////////////////////////////////////////////////////

typedef struct MODULUS2D{
	double xxxx;
	double yyyy;
	double xxxy;
	double yyyx;
	double xyxy;
	double xxyy;
	double reg;
}modulus2D;


typedef struct EIGEN_FIELD{
	int dim;
	int field_size;
	double** Eig;
	double** Vec;
}eigen_field;

// constants //////////////////////////////////////////////////////

// simulation
const int dimension = 2;
const int degrees_of_freedom = F_PRIM;
const int add_field_num = F_SEC;
const int global_size = F_GLOB_VAR;
const int Crack_constrained = YES;
const int constraint_solver = REFLECTIVE_NEWTON;
const int strain_decomposition = TRACE;
const int n_u = 0;
const int n_h = 0;
const int n_c = 0;
const int n_m = 1;
const int n_T = 0;

const double damage_zero_slope = 1.0;					// slope g'(z=0) of degradation factor
const double crack_thres = 0.20;						// below this value, material is considered as cracked and the mesh is refined
														// tol=0: effect disabled
const double Boltz = 5.67e-14;							// S. Boltzmann constant in W/(mm^2*K^4)
const double R_universal = 8.314;						// Universal gas constant in J/(mol K)
const double constraint_shift = -0.1;					// prevent initial damage variable to be loacted at the bounds, shift down by 10%
const double T_absolute_ref = 273.0;					// Temperature reference 0 °C
const double T_ref_mobility = 398;						// in K, mobility refers to that timeperature if Arrhenius law is used
const double minimum_c = -1.;							// minimum constraint for chemical field c
const double maximum_c = 1.;							// maximum constraint for chemical field c
																												
const char* Bin_filename = "data.bin";
const char* Bin_finalname = "final.bin";
const char* Default_param_name = "parameters.params";

const double front_resolution = 10.;						// gradient refinement: amount of nodes resolving a front
const int histogram_resolution = 100;					// resolution of histogram

// mesh generator
const double min_edge_len = 0.001;
const double edge_refine_fraction = 0.5;
							
							
														
// global variables ////////////////////////////////////////////////

/* Solution organization

	0. u x-component
	1. u y
	2. h
	3. c
	4. mu 
	5. T
*/

// input/output
static char Param_dir[512];
static char Output_name[512];
static char Output_dir[512];
static char Mesh_name[512];
static char Mesh_dir[512];
static char Cond_name[512];
static char Load_name[512];							// full path
static char* Home_dir;

static int load_mode = NO;
static int load_index = LAST_INDEX;

// model parameters

static double mu_1 = 1.0;			
static double mu_2 = 1.0;		
static double lambda_1 = 1.0;
static double lambda_2 = 1.0;	

static modulus2D Moduli_1 = {.xxxx=0,.yyyy=0,.xxxy=0,.yyyx=0,.xxyy=0,.xyxy=0,.reg=0};
static modulus2D Moduli_2 = {.xxxx=0,.yyyy=0,.xxxy=0,.yyyx=0,.xxyy=0,.xyxy=0,.reg=0};
static modulus2D* Elastic_moduli = NULL;

static double a_1 = 1.0;
static double a_2 = 1.0;
static double a_interface = 1e0;
static double interface_thougness_min = 1e-2;		// if alpha shall be reduced at domain borders
static double beta = 1.0;
static double kappa = 1.0;
static double xi = 0.0;
static double eta_eps = 1e-3;
static double energy_tol = 0;
static double margin_large = 0.2;					// assume large thougness at margon to prevent cracking at Dirichlet bounds
static double large_toughness = 100.;					
static point2D Robin_stiffness = {.x=1e+03,.y=1e+03};
static matrix2D thermal_expansion1 = {.xx=0.01,.xy=0,.yx=0,.yy=0.01};
static matrix2D thermal_expansion2 = {.xx=0,.xy=0,.yx=0,.yy=0};
static matrix2D interface_tensor1 = {.xx=0.625e-6,.xy=0.375e-6,.yx=0.375e-6,.yy=0.625e-6};
static matrix2D interface_tensor2 = {.xx=0.625e-6,.xy=0.375e-6,.yx=0.375e-6,.yy=0.625e-6};

static double kappa_c = 0.01;
static double mobility1 = 1.;
static double mobility2 = 0;
static double mobility_exp1 = 7.675e+04;
static double mobility_exp2 = 0;

static double vis_c = 0.0;
static double pot_factor = 1.0;
static double c_eutectic = 2.*0.5-1.;

static double T_crit = 456.0;						// in Kelvin
static double T_ref = 183.0;						// in Celcius
static double T_outer = 0.0;						// in Celcius
static double kappa_T1 = 0.01;
static double kappa_T2 = 0.1;
static double kappa_T0 = 0.01;
static double kappa_bound = 5.0;
static double rho_1 = 1.;
static double rho_2 = 1.;
static double rho_0 = 1.;
static double surface_emissivity = 0.0;//0.95;

// discretization
static int backtracking = NO;
static int time_adaptive = NO;
static int space_adaptive = YES;
static double min_refine_area = 5e-5;
static double max_refine_area = 1.;
static double strain_ref_alpha_factor = 1e-1;			// fraction of critical strain value (alpha)
static double strain_ref_max_area = 5e-4;				// maximum fraction of total area that con be refined
static long time_steps = 1e3;
static double total_time = 1.;
static double dt = 1e-3;
static int sample_interval = 1;
static double time_adaptive_threshold_dH = 0.01;
static double time_adaptive_min_dt = 0.01;
//static double time_adaptive_max_dt = 0.01;
static double eps_constraint = 3e-4;
static double time_offset = 0;
static double pure_neumann_reg_factor = 1e-6;			// elastic regularization for pure Neumann conditions

// interpolation
static double interpolation_tol = 1e-8;
static int interpolation_method = ELEMENT_BASE;

// Alternate Minimization
static int Alt_info = YES;
static int Alt_trust_region = NO;
const int trust_control_radius = FREE_EXPAND;
const int trust_control_rejection = PARTIAL_REJECTION;
static int Alt_max_iter = 400;
static double Alt_eps = 1e-8;
const double Alt_trust_decrease = 0.2;
static double Alt_trust_min = 1e-15;
static double Alt_trust_max = 1e4;
static double Alt_trust_thres_min = 0.1;
static double Alt_trust_thres_max = 0.8;

// working matrices
static sparse_matrix3D* Strain_grad = NULL;
static sparse_matrix3D* Strain_abs = NULL;
static sparse_matrix3D* Strain_grad_bound = NULL;
static sparse_matrix3D* Phase_Jacobian = NULL;
static sparse_matrix3D* T_Flux = NULL;
static sparse_matrix3D* T_Source = NULL;
static sparse_matrix3D* Jacobian_Ux = NULL;
static sparse_matrix3D* Jacobian_Uy = NULL;
//static sparse_matrix* Interface_damage = NULL;
static sparse_matrix* Damage_Hesse_linear = NULL;
//static sparse_matrix* Damage_Hesse_nonlinear = NULL;
static sparse_matrix* Regularization = NULL;
static sparse_matrix* Bound_Dirichlet_reg = NULL;
static sparse_matrix* Bound_Neumann_reg = NULL;
static sparse_matrix* Bound_Neumann = NULL;
static sparse_matrix* Bound_Active = NULL;

static sparse_matrix* Interface_chemical = NULL;
static sparse_matrix* ID_damage = NULL;
static sparse_matrix* Phase_Diffusion_c = NULL;
static sparse_matrix3D* Phase_Diffusion_mu = NULL;
static sparse_matrix* Phase_ID_c = NULL;
static sparse_matrix* Phase_ID_mu = NULL;
static sparse_matrix* Phase_ID_cross = NULL;
static sparse_matrix* T_Flux_Reg = NULL;
static sparse_matrix* T_Capacity = NULL;
static sparse_matrix* T_Neumann = NULL;
static sparse_matrix* Dissipation = NULL; 
static sparse_matrix* Strain_tensor_Ax = NULL;
static sparse_matrix* Strain_tensor_Ay = NULL;
static sparse_matrix* Interpolation = NULL;

static map_sparse* El_Hesse_Map = NULL;


// working arrays
static double** All_U = NULL;					// displcament field
static double** All_H = NULL;					// damage field (scalar)
static double** All_C = NULL;					// phase field (phase1:  c=1, phase2: c=-1)
static double** All_T = NULL;					// temperature

static eigen_field* Glob_eigensystem0 = NULL;
static int* Glob_mesh_sizes = NULL;
static int attribute_num = 0;
static double* Attributes = NULL;
static double* Glob_Pointer_C = NULL;
static double* Glob_Pointer_H0 = NULL;
static double* Glob_Pointer_Trace0 = NULL;
static double* Strain_energy = NULL;
static double* dissipation_damage = NULL;
static double* dissipation_chemical = NULL;
static double* heat_bound_flux = NULL;
static double* energy_constr_lower = NULL;
static double* energy_constr_upper = NULL;
static double* entropy_production = NULL;
static double* Times = NULL;

static int SIGINT_flag = 0;
static int glob_mesh_index = 0;
static int glob_time_index = 0;
static double glob_time = 0;
static char* Prog_name;
static mesh2D base_mesh;
static element_collection base_elements;
static init_info Initial_conditions;
static bound_info* Boundary_conditions[F_PRIM];
static double free0 = 0;
static double entropy0 = 0;
static double inner0 = 0;
static double dt_min;
static double dt_max;

//Reflective Newton
static int* R_Newton_Condition = NULL;
static int R_Newton_max_iter = 200;
static double R_Newton_eps = 1e-8;
static double R_Newton_sigma_l = 0.1;
static double R_Newton_sigma_u = 0.9;
static double R_Newton_rho = 0.1;
static double R_Newton_Delta = 250.;					// sqrt(n)/2 good choice
static double R_Newton_Xi = 1.;
static double R_Newton_trust_lower = 0.6;
static double R_Newton_trust_upper = 1.1;
static double R_Newton_trust_radius_min = 1e-3;
static double R_Newton_trust_radius_max = 1000.;
static double R_Newton_Dir_tol = 1e-6;

// for AMG solver 
//static amg_system_info* El_AMG = NULL;
//static amg_system_info* Int_AMG = NULL;
//static amg_system_info* Diff_AMG = NULL;

static amg_statistics step1_AMG_stat;
static amg_statistics alter_min_stat;

// BiCGStab
//static double BiCGStab_eps = 1e-3;			// residuum reduction in each BiCGStab step
//static int BiCGStab_max_iter = 10;			// maximum iteration number

// AMG
//static int AMG_smoothing_iter = 2;

/*static int step1_AMG_depth = 5;
static int step1_max_iter = 100;			// maximum namber of V cycles in AMG
static double step1_eps = 1e-8;				// residuum reduction for each preconditer step
static double step1_tol = 1e-2;				// elements smaller than this treshold are removed in Prolongation/Restriction

static int step2_AMG_depth = 4;
static int step2_max_iter = 100;			// same for preconditioner in step 2
static double step2_eps = 1e-3;				
static double step2_tol = 1e-1;		*/

// for direct solver (UMFPACK, Multifrontal)
static double* UMFPACK_pattern_options = NULL;
static double* UMFPACK_factor_options = NULL;
static double* UMFPACK_solve_options = NULL;

static double UMFPACK_pattern_info[UMFPACK_INFO];	
static double UMFPACK_factor_info[UMFPACK_INFO];	
static double UMFPACK_solve_info[UMFPACK_INFO];	

static double min_cond_num = INFINITY;
static double max_cond_num = 0;
static double UMFPACK_time_damage = 0;
static double UMFPACK_time_elastic = 0;
static double UMFPACK_time_chemistry = 0;
static double UMFPACK_time_temperature = 0;
static double UMFPACK_time_interpolation = 0;

static FILE* Cond_file = NULL;

static int omp_proc_num = 1;

// function declarations /////////////////////////////////////////////////////////

void test_AMG();
double linear_limited(double c);
double linear_limited_derivative(double c);
double linear_limited_derivative2(double c);
double* interpolated(double* X,int dim,int size_per_dim,sparse_matrix* LS_Matrix,sparse_matrix* RS_Matrix);
sparse_matrix* Get_interpolation_matrix(mesh2D* New_mesh,mesh2D* Old_mesh,element_collection* New_elements,element_collection* Old_elements);
void set_glob_mesh(mesh2D* Mesh,element_collection* Triangles);
void Load_mesh(int index,mesh2D* Mesh,element_collection* Triangles,int quiet);
void read_mesh_2D(char* Dir,char* Name,mesh2D* Mesh,element_collection* Triangles,double** Attributes,int* attr_num,int quiet);
void refine_by_gradient(double* X,double* Refine,int field_number,double a_min,double a_max);
double* compute_strain_energy(double* U,double* H,double* C,double* T,int order_dz);
double* compute_divergence(double* X);
void test_rank4_symmetries(double* C);
double* random_initial(double mean,double var,int size);
double* Two_phase_interpolation(double factor1,double factor2,double* C,int n);
double* Self_strain_energy(double* U,double* H,double* C,double* Th);
double* entropy(double* U,double* H,double* C,double* T);
double* damage_predictor(double* U,double* H,double* C,double* T);
double* damage_gradient(double* X);
double* damage_rate_independent(double* C);
double* linear_strain_tensor(double* U);
double* elastic_constitutive(double* U,double* H,double* C,double* T,int order_dz,int order_dc);
double* elastic_potential(double* U,double* H,double* C,double* T,int order_dz,int order_dc);
sparse_matrix* elastic_potential_Hesse(double* U,double* H,double* C,double* T,int order_dz,int order_dc);
double* free_energy_heat_dT(double* C,double* T);
double* total_strain(double* Strain,double* C,double* T,int order_dc,int order_dT);
double* secondorder_predictor(double* U,double* H,double* C,double* T);
double* get_constraint_force(double* U,double* H,double* H_prev,double* C,double* T);
double* max_steepnes_2D(double* X,double* Weight,int field_number);
void get_crack_set(double* H,double tol,int** Nodes,int* size);
double Free_energy_chemical(double* C,double* T);
double Free_energy_elastic(double* U,double* H,double* C,double* T);
double Total_free_energy(double* U,double* H,double* C,double* T);
double* Heat_source(double* U,double* U_prev,double* H,double* H_prev,double* C,double* C_prev,double* Mu,double* T);
double* heat_capacity_field(double* U,double* H,double* C,double* T);
double* damage_driving_force(double* H,double* H_prev,double* C);
double* damage_interface_tensor(double* H,double* C,int order_dc);
double* Deformation_gradient(double* U);
double* neighbour_smoother(double* X);
double* Metric(double* U);
double* first_invariant(double* G);
double* second_invariant(double* G);
double* third_invariant(double* G);

void Test_AMG(sparse_matrix* A);

// functions mesh mode /////////////////////////////////////////////////////////////////////

double F_area_disc(point2D* P){
	const double y_min = -1.;
	const double y_max = 1.;
	const double val_min = 0.001;
	const double val_max = .5;
	double alpha = -M_PI/4.5;
	double y = cos(alpha)*P->y-sin(alpha)*P->x;
	return val_min+(val_max-val_min)/(y_max-y_min)*(y-y_min);
}

double F_area_shape6(point2D* P){
	const double val_min = 1e-4;
	const double val_max = 1.;
	const double l = 1.;
	const double f = 0.2;
	const double d = 0.05;
	const double aspectratio = 1.;
	
	double s = 2.*d;
	point2D p = init_point2D(0,l-f);
	double r = sqrt((P->x-p.x)*(P->x-p.x)+(P->y-p.y)*(P->y-p.y));
	
	double res = val_max;
	res -= (val_max-val_min)*exp(-r*r/(2.*s*s));
	return res;
}

double F_area_bound_curvature(point2D* P){
	const double fac = 0.25;
	
	static int* Bound_index = NULL;
	static int* Bound_elements = NULL;
	static double* Coeff = NULL;
	static int bound_size = 0;
	static int bound_el_size = 0;
	
	int i,j,k;
	int n = glob_mesh.size;
	
	if (P==NULL){																	// init
		get_boundary_elements(&Bound_elements,&bound_el_size);
		for (i=0;i<n;i++){
			if (is_boundary(i)>=0){
				bound_size++;
				Bound_index = (int*)realloc(Bound_index,bound_size*sizeof(int));
				Bound_index[bound_size-1] = i;
			} 
		}
		double curv;
		Coeff = zero_vector(n);
		for (i=0;i<bound_size;i++){
			k = Bound_index[i];
			curv = bound_curvature(k);
			if (curv==0) Coeff[k] = 1.;
			Coeff[k] = fac/fabs(curv);
			if (Coeff[k]>1.) Coeff[k] = 1.;
		}
		//print_vector(Coeff,n);
		return 0;
	}
	else{				
		index3D ind;
		point2D bary;		 														// construct area function
		if (Bound_index!=NULL && Bound_elements!=NULL){
			double res = 0;
			for (i=0;i<bound_el_size;i++){
				ind = elements.Elements[Bound_elements[i]];
				if (in_triangle(&(glob_mesh.Points[ind.i]),&(glob_mesh.Points[ind.j]),&(glob_mesh.Points[ind.k]),P)==1){
					get_barycentric_2D(&(glob_mesh.Points[ind.i]),&(glob_mesh.Points[ind.j]),&(glob_mesh.Points[ind.k]),P);
					res = bary.x*Coeff[ind.i]+bary.y*Coeff[ind.j]+(1.-bary.x-bary.y)*Coeff[ind.k];
					break;
				}
			}
			return res;			
		}
		else{
			printf("function F_area_bound_curvature not initialized -> abort\n");
			exit(0);
		}
	}
}


double create_2D_mesh(char* name, int bound_nodes){
	point2D* Points = (point2D*)malloc(bound_nodes*sizeof(point2D));
	double area = create_arc(bound_nodes,1.,2.*M_PI,Points);	
	create_poly_file(name,Points,bound_nodes);
	free(Points);
	return area;
}

double eq_distr(int i,int size){
	return (double)i/size;
}

double tanh_distr(int i,int size){
	const double a = 8.;
	return (1.+tanh(a*((double)i/size-0.5)))/2.;
}

void add_line_to_Poly(point2D** Poly,int* len,point2D* P1,point2D* P2,int num,double (*Distr)(int i,int size)){
	int i,start;
	double x,y,s;
	if (P1==NULL){
		P1 = &((*Poly)[*len-1]);
		start = 1;
	}
	else start = 0;
	
	*Poly = (point2D*)realloc(*Poly,(*len+num+1-start)*sizeof(point2D));
	
	for (i=start;i<=num;i++){
		s = (*Distr)(i,num);
		x = P1->x+(P2->x-P1->x)*s;
		y = P1->y+(P2->y-P1->y)*s;
		(*Poly)[*len+i-start] = init_point2D(x,y);
	}
	*len += num+1-start;
}

void line_close_Poly(point2D** Poly,int* len,point2D* P2,int num,double (*Distr)(int i,int size)){
	int i,start;
	double x,y,s;
	point2D* P1 = &((*Poly)[*len-1]);
	
	start = 1;
	*Poly = (point2D*)realloc(*Poly,(*len+num+1-start)*sizeof(point2D));
	for (i=start;i<=num-1;i++){
		s = (*Distr)(i,num);
		x = P1->x+(P2->x-P1->x)*s;
		y = P1->y+(P2->y-P1->y)*s;
		(*Poly)[*len+i-start] = init_point2D(x,y);
	}
	*len += num-start;
}

void add_arc_to_Poly(point2D** Poly,int* len,point2D* P,point2D* Center,double angle,int num,double (*Distr)(int i,int size)){
	int i;
	double x,y,s;
	if (P==NULL) P = &((*Poly)[*len-1]);
	*Poly = (point2D*)realloc(*Poly,(*len+num)*sizeof(point2D));
	point2D R = vec_diff(P,Center);

	for (i=1;i<=num;i++){
		s = (*Distr)(i,num);
		x = cos(s*angle)*R.x-sin(s*angle)*R.y;
		y = sin(s*angle)*R.x+cos(s*angle)*R.y;
		(*Poly)[*len+i-1] = init_point2D(Center->x+x,Center->y+y);
	}
	*len += num;
}

void add_circle_to_Poly(point2D** Poly,int* len,point2D* Center,double r,int num){
	
	double angle = 2.*M_PI*(1.-(double)1./num);
	point2D* P = clone_point(Center);
	P->x += r;
	if ((*len)!=0)add_line_to_Poly(Poly,len,NULL,P,1,&eq_distr);
	else{
		*Poly = (point2D*)malloc(sizeof(point2D));
		(*Poly)[0] = init_point2D(P->x,P->y);
		(*len)++;
	}
	add_arc_to_Poly(Poly,len,P,Center,angle,num,&eq_distr);
	free(P);
}

void add_ellipse_to_Poly(point2D** Poly,int* len,point2D* Center,double a,double b,int num,double (*Distr)(int i,int size)){
	
	const double power = 1.;//0.5;
	
	int i;
	double x,y,s,angle;
	
	
	point2D* P = clone_point(Center);
	P->x += a;
	if ((*len)!=0) add_line_to_Poly(Poly,len,NULL,P,1,&eq_distr);
	else{
		*Poly = (point2D*)malloc(sizeof(point2D));
		(*Poly)[0] = init_point2D(P->x,P->y);
		(*len)++;
	}
	
	*Poly = (point2D*)realloc(*Poly,(*len+num)*sizeof(point2D));
	angle = 2.*M_PI*(1.-(double)1./num);
	for (i=1;i<=num;i++){
		s = (*Distr)(i,num);
		x = cos(s*angle)>=0 ? pow(fabs(cos(s*angle)),power)*a : -pow(fabs(cos(s*angle)),power)*a;
		y = sin(s*angle)*b;
		(*Poly)[*len+i-1] = init_point2D(Center->x+x,Center->y+y);
	}
	*len += num;
	
	free(P);
}

void add_notch_to_poly(point2D** Poly,int* len,point2D* Start,point2D* End,point2D* Tip,double width,int line_num,int circ_num){
				
		point2D* C = clone_point(Start);
		vec_add_mult(C,End,1.);
		vec_mult(C,0.5);				
		point2D* D = clone_point(End);
		vec_add_mult(D,C,-1.);
		normalize(D);
				
		point2D* F = clone_point(Tip);
		vec_add_mult(F,C,-1.);
		point2D* N = clone_point(F);
		normalize(N);
		vec_add_mult(F,N,-width/2.);
		vec_add_mult(F,C,1.);
		
		point2D w = {.x=-N->y,.y=N->x};
		if (vec_scalar(&w,D)<0) vec_mult(&w,-1.);
		else if (vec_scalar(&w,D)==0){
			printf("notch parallel to boundary -> abort\n");
			exit(0);
		}
		point2D* P1 = clone_point(F);
		vec_add_mult(P1,&w,-width/2.);
		point2D* P2 = clone_point(F);
		vec_add_mult(P2,&w,width/2.);
								
		add_line_to_Poly(Poly,len,NULL,P1,line_num,&eq_distr);
		add_arc_to_Poly(Poly,len,P1,F,M_PI,circ_num,&eq_distr);	
		add_line_to_Poly(Poly,len,NULL,End,line_num,&eq_distr);
		
		free(C);
		free(D);
		free(F);
		free(N);
		free(P1);
		free(P2);
}		

double create_shape1(char* name,int mesh_size){
	const int n = 7;
	const double l = 1.;
	const double a = 0.2;
	const double h = 0.4;
	
	int i;
	point2D* Points = (point2D*)malloc(n*sizeof(point2D));
	
	Points[0] = init_point2D(-l,0);											//	---\  /---
	Points[1] = init_point2D(l,0);											//	|	\/   |
																			//	|		 |
	Points[2] = init_point2D(l,l);											//  |________|
	Points[3] = init_point2D(a*l,l);										
																			
	Points[4] = init_point2D(0,h*l);
	Points[5] = init_point2D(-a*l,l);
	
	Points[6] = init_point2D(-l,l);
	
	int poly_size = n;
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape1a(char* name,int mesh_size){
	const int m = 50;
	const double l = 1.;
	const double aspectratio = 0.5;
	const double d = l/5.;
	const double w = l/10.;
	
	int i;
	double s;
	point2D P1,P2;
	int poly_size = 0;
	point2D* Points = NULL;
	P1 = init_point2D(-l/(2.*aspectratio),0);										//	 ___  ___
	P2 = init_point2D(-l/(2.*aspectratio),l);										//	|   \/   |
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);						//	|		 |
	P2 = init_point2D(-d,l);														//  |___/\___|
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(0,(l+w)/2.);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,m,&eq_distr);
	P2 = init_point2D(d,l);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,m,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),l);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),0);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(d,0);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(0,(l-w)/2.);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,m,&eq_distr);
	P2 = init_point2D(-d,0);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,m,&eq_distr);
	
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape2(char* name,int mesh_size){
	const int n = 4;
	const int m = 400;
	const double l = 1.;
	const double aspectratio = 1.;
	
	int i;
	double s;
	point2D P1,P2;
	//int poly_size = 2*m+2;//(m>0) ? 6*m : 4;
	//point2D* Points = (point2D*)malloc(poly_size*sizeof(point2D));
	
	int poly_size = 0;
	point2D* Points = NULL;
	P1 = init_point2D(-l/(2.*aspectratio),0);										//	----------
	P2 = init_point2D(-l/(2.*aspectratio),l);										//	|	     |
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);						//	|		 |
	P2 = init_point2D(l/(2.*aspectratio),l);										//  |________|
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),0);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	//add_line_to_Poly(&Points,&poly_size,NULL,&P1,1,&eq_distr);
	
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	//edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape3(char* name,int mesh_size){
	const double l = 1.;
	const double r = 0.15;
	const double a = 0.1;
	const double b = 0.2;
	const double aspectratio = 1.;
	const double d = 0.01;
	const double f = 0.3;
	const int hole_num = 1;
	const int row_num = 1;
	const int m = 200;
	const int q = 100;
	const int s = 50;
	const point2D offset = {.x=0.0,.y=0};//{.x=-0.25,.y=-0.15};
	int i,j,k,old;
	double x,y;
	point2D center,P1,P2,Tip;
	
	point2D* Points = NULL;
	point2D* Outer = (point2D*)malloc(hole_num*row_num*sizeof(point2D));
	int* Sizes = (int*)malloc((hole_num*row_num+1)*sizeof(int));
	
	
	int poly_size = 0;																//   ________
	P1 = init_point2D(-l/(2.*aspectratio),0);										//	|    |   |
	P2 = init_point2D(-l/(2.*aspectratio),l);										//	|   /\   |
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);						//	|	\/ 	 |
	/*P1 = init_point2D(-d/2.,l);													//  |________|
	P2 = init_point2D(d/2.,l);	
	Tip = init_point2D(0,l-f);
	add_line_to_Poly(&Points,&poly_size,NULL,&P1,1,&eq_distr);
	add_notch_to_poly(&Points,&poly_size,&P1,&P2,&Tip,d,q,s);*/
	P2 = init_point2D(l/(2.*aspectratio),l);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),0);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	line_close_Poly(&Points,&poly_size,&P1,1,&eq_distr);
	Sizes[0] = poly_size;
	
	for (j=0;j<row_num;j++){
		for (i=0;i<hole_num;i++){
			old = poly_size;					
			x = (double)l*(i+1)/(hole_num+1)/(aspectratio)-1./(2.*aspectratio)+offset.x;
			y = (double)l*(j+1)/(row_num+1)+offset.y;
			Outer[j*hole_num+i] = init_point2D(x,y);
			center = init_point2D(x,y);
			//add_circle_to_Poly(&Points,&poly_size,&center,r,m);
			add_ellipse_to_Poly(&Points,&poly_size,&center,a,b,m,&eq_distr);
			Sizes[j*hole_num+i+1] = poly_size-old;
		}
	}
	
	int total_size = 0;
	int* Loop_start = zero_int_list(hole_num*row_num+2);
	for (k=1;k<hole_num*row_num+2;k++){
		total_size += Sizes[k-1];
		Loop_start[k] = total_size;
	}
	
	edge_refine(&Points,Loop_start,hole_num*row_num+1,M_PI/6.,min_edge_len,edge_refine_fraction);
	
	point2D** Polys = (point2D**)malloc((hole_num*row_num+1)*sizeof(point2D*));
	for (k=0;k<hole_num*row_num+1;k++){
		Sizes[k] = Loop_start[k+1]-Loop_start[k];
		Polys[k] = (point2D*)malloc(Sizes[k]*sizeof(point2D));		
		for (j=Loop_start[k];j<Loop_start[k+1];j++) Polys[k][j-Loop_start[k]] = init_point2D(Points[j].x,Points[j].y);
	}
	
	create_nonconvex_poly_file(name,Polys,Sizes,hole_num*row_num+1,Outer);
	
	for (k=0;k<hole_num*row_num+1;k++) free(Polys[k]);
	free(Polys);
	free(Loop_start);
	free(Points);
	free(Sizes);
	free(Outer);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape4(char* name,int mesh_size){
	const int n = 4;
	const double l = 1.;
	const double rx = 0.02;
	const double ry = 0.4;
	
	int i;
	double a;
	int m = 4*round(0.8*M_PI*ry/l*sqrt((double)mesh_size));
	m = 200;
	if (m % 2 == 0) m++;
	int poly_size = 2*m+6;
	point2D* Points = (point2D*)malloc(poly_size*sizeof(point2D));
	
	Points[0] = init_point2D(-l,0);											
	for (i=0;i<=m;i++){
		a = (double)M_PI*(m-i)/m;
		Points[i+1] = init_point2D(cos(a)*rx,sin(a)*ry);						//   __	   __
	}																			//	|  |__|  |
	Points[m+2] = init_point2D(l,0);											//	|	__   |																		//	|		 |
	Points[m+3] = init_point2D(l,l);											//  |__|  |__|
	for (i=0;i<=m;i++){
		a = (double)-M_PI*i/m;
		Points[i+m+4] = init_point2D(cos(a)*rx,l+sin(a)*ry);
	}
	Points[2*m+5] = init_point2D(-l,l);
	
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape5(char* name,int mesh_size){
	const int n = 8;
	const double l = 1.;
	
	int i;
	point2D* Points = (point2D*)malloc(n*sizeof(point2D));
	
	Points[0] = init_point2D(0,0);											//	_______
	Points[1] = init_point2D(l,0);											//	|	   |
	Points[2] = init_point2D(l,l);											//	|	   |_
	Points[3] = init_point2D(l/2.,l);										//  |_       |
	Points[4] = init_point2D(l/2.,3.*l/2.);									//    |      |
	Points[5] = init_point2D(-l/2.,3.*l/2.);								//	  |______|
	Points[6] = init_point2D(-l/2.,l/2.);
	Points[7] = init_point2D(0,l/2.);
	
	int poly_size = n;
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape6(char* name,int mesh_size){
	const int n = 4;
	const double l = 1.0;
	const double f = 0.1;
	const double d = 0.003;
	const double aspectratio = 1.0;
	
	int i;
	double a;
	int m = 21;
	int poly_size = m+7;
	point2D* Points = (point2D*)malloc(poly_size*sizeof(point2D));
	
	Points[0] = init_point2D(-l/(2.*aspectratio),0);							//  ________
	Points[1] = init_point2D(-l/(2.*aspectratio),l);							// |	|	|
	Points[2] = init_point2D(-d/2.,l);											// |	|	|
	for (i=0;i<=m;i++){															// |________|
		a = (double)M_PI*i/m;
		Points[i+3] = init_point2D(-cos(a)*d/2.,l-f-sin(a)*d/2.);				
	}	
	Points[m+4] = init_point2D(d/2.,l);				
	Points[m+5] = init_point2D(l/(2.*aspectratio),l);	
	Points[m+6] = init_point2D(l/(2.*aspectratio),0);	
					
	/*Points[0] = init_point2D(-l,0);	
	Points[1] = init_point2D(-l,l/2.-d/2.);	
											
	for (i=0;i<=m;i++){
		a = (double)M_PI*i/m;
		Points[i+2] = init_point2D(sin(a)*d/2.,l/2.-cos(a)*d/2.);				//   ________	  
	}																			//	|  	     |
	Points[m+3] = init_point2D(-l,l/2.+d/2.);									//	|----    |																		//	|		 |
	Points[m+4] = init_point2D(-l,l);											//  |________|
	Points[m+5] = init_point2D(l,l);
	Points[m+6] = init_point2D(l,0);*/
	
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape6a(char* name,int mesh_size){
	const double l = 1.;
	const double f = 0.2;
	const double d = 0.01;
	const double aspectratio = 1.;
	
	int i;
	double a;
	point2D P1,P2;
	
	int m = 20;
	int n = 200;
	int k = 400;
	
	int poly_size = 0;
	point2D* Points = NULL;
	P1 = init_point2D(-l/(2.*aspectratio),0);
	P2 = init_point2D(-l/(2.*aspectratio),l);
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,k,&eq_distr);
	P2 = init_point2D(-d/2.,l);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,n,&eq_distr);
	P2 = init_point2D(-d/2.,l-f);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,n,&eq_distr);
	P2 = init_point2D(0,l-f);
	add_arc_to_Poly(&Points,&poly_size,NULL,&P2,M_PI,m,&eq_distr);
	P2 = init_point2D(d/2.,l);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,n,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),l);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,n,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),0);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,k,&eq_distr);
	//add_line_to_Poly(&Points,&poly_size,NULL,&P1,2*k,&eq_distr);
	
	//print_point_list2D(Points,poly_size,"points");
	//exit(0);
	 
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	//edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape7(char* name,int mesh_size){
	const double l = 1.;
	const double beta = M_PI/2.*(1.-0.3);
	
	int i;
	double a,x,y;
	int m = 30;
	int poly_size = 2*(2*m+1);
	point2D* Points = (point2D*)malloc(poly_size*sizeof(point2D));
										
	for (i=-m;i<=m;i++){
		a = (double)(M_PI/2.-beta)*i/m;
		x = l*(1.-tan(beta)/2.+cos(a)/(2.*cos(beta)));
		y = l*(1.+sin(a)/cos(beta))/2.;
		Points[i+m] = init_point2D(-x,y);						
		Points[3*m+1+i] = init_point2D(x,l-y);									//	 ______	
	}																			//	/      \
																				//  |	   |																		//	|		 |
																				//  \______/
	int* Loop_start = zero_int_list(2);
	Loop_start[1] = poly_size;
	edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
	poly_size = Loop_start[1];
	free(Loop_start);
	
	create_poly_file(name,Points,poly_size);
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape8(char* name,int mesh_size){
	//const int n = 4;
	const int m = 400;
	const double l = 1.;
	const double aspectratio = 0.5;
	const int rows = 1;
	const int cols = 2;
	const double a = 0.3;
	const double b = 0.4;
	
	int i,j,k,old;
	double s;
	point2D P1,P2,center;
	
	int poly_size = 0;
	point2D* Points = NULL;
	P1 = init_point2D(-l/(2.*aspectratio),0);										//	 ________
	P2 = init_point2D(-l/(2.*aspectratio),l);										//	|    _   |
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);						//	|	|_|  |
	P2 = init_point2D(l/(2.*aspectratio),l);										//  |________|
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),0);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	line_close_Poly(&Points,&poly_size,&P1,1,&eq_distr);
	
	int* Sizes = zero_int_list(rows*cols+1);
	point2D* Outer = (point2D*)malloc(rows*cols*sizeof(point2D));
	Sizes[0] = poly_size;
	for (i=0;i<rows;i++){
		for (j=0;j<cols;j++){
			old = poly_size;
			point2D br = init_point2D(-l/(2.*aspectratio)+(double)(j+1)*l/(aspectratio*(cols+1))-a/2.,(double)(i+1)*l/(rows+1)-b/2.);			
			P1 = init_point2D(br.x+a,br.y);
			add_line_to_Poly(&Points,&poly_size,&br,&P1,1,&eq_distr);
			P1 = init_point2D(br.x+a,br.y+b);
			add_line_to_Poly(&Points,&poly_size,NULL,&P1,1,&eq_distr);
			P1 = init_point2D(br.x,br.y+b);
			add_line_to_Poly(&Points,&poly_size,NULL,&P1,1,&eq_distr);
			line_close_Poly(&Points,&poly_size,&br,1,&eq_distr);
			Outer[i*cols+j] = init_point2D(br.x+a/2.,br.y+b/2.);
			Sizes[i*cols+j+1] = poly_size-old;
		}
	}
	
	int total_size = 0;
	int* Loop_start = zero_int_list(rows*cols+2);
	for (k=1;k<rows*cols+2;k++){
		total_size += Sizes[k-1];
		Loop_start[k] = total_size;
	}
	
	edge_refine(&Points,Loop_start,rows*cols+1,M_PI/6.,min_edge_len,edge_refine_fraction);
	
	point2D** Polys = (point2D**)malloc((rows*cols+1)*sizeof(point2D*));
	for (k=0;k<rows*cols+1;k++){
		Sizes[k] = Loop_start[k+1]-Loop_start[k];
		Polys[k] = (point2D*)malloc(Sizes[k]*sizeof(point2D));		
		for (j=Loop_start[k];j<Loop_start[k+1];j++) Polys[k][j-Loop_start[k]] = init_point2D(Points[j].x,Points[j].y);
	}

	create_nonconvex_poly_file(name,Polys,Sizes,rows*cols+1,Outer);
	
	for (k=0;k<rows*cols+1;k++) free(Polys[k]);
	free(Polys);
	free(Loop_start);
	free(Points);
	free(Sizes);
	free(Outer);
	
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_shape_hexagons(char* name,char* condname,int mesh_size){
	
	const int N = 1; //12
	const int M = 1;  //3
	const double l = 1.;
	const double a = (double)l/(3*N);
	const double fac = 0.65;
	const double h = sqrt(3.)*a/2.;
	const double r = (1.-fac)*h;
	const double q = r/sqrt(3.);
	
	int i,j,k,old;
	double sx,sy;
	point2D* Q;
	point2D P1,P2,last,center,mean;
	
	int poly_size = 0;
	point2D* Points = NULL;
	
	point2D* Bound_i = NULL;
	point2D* Bound_o = NULL;
	if (condname!=NULL){
		Bound_i = (point2D*)malloc((2*N*M)*sizeof(point2D));
		Bound_o = (point2D*)malloc(4*sizeof(point2D));		
	}
		
	P1 = init_point2D(-a/2.-r,h);				
	
	mean = init_point2D(P1.x,P2.x);
	for (i=0;i<N;i++){		
		sx = (double)3*i*a;			
		P2 = init_point2D(sx-q,-r);	
		vec_add_mult(&mean,&P2,1.);							
		if (i==0) Q = &P1; else Q = NULL;
		add_line_to_Poly(&Points,&poly_size,Q,&P2,1,&eq_distr);						
		P2 = init_point2D(sx+a+q,-r);	
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		P2 = init_point2D(sx+3.*a/2.+q,h-r);	
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		P2 = init_point2D(sx+5.*a/2.-q,h-r);	
		if (i==N-1) P2.x += 2.*q;
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
	}
	if (condname!=NULL) Bound_o[0] = P2;
	
	sx += 5.*a/2.;
	for (j=0;j<M;j++){		
		sy = (double)(2*j+1)*h;		
		P2 = init_point2D(sx+a/2.+r,sy+h);
		vec_add_mult(&mean,&P2,1.);		
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		if (j<M-1){
			P2 = init_point2D(sx+r,sy+2.*h);
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
			vec_add_mult(&mean,&P2,1.);	
		}
	}
	if (condname!=NULL) Bound_o[1] = P2;
	
	sy += 2.*h;
	for (i=0;i<N;i++){		
		sx = (double)3*(N-1-i)*a+5.*a/2.;			
		P2 = init_point2D(sx+q,sy+r);	
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		P2 = init_point2D(sx-a-q,sy+r);										
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);						
		P2 = init_point2D(sx-3.*a/2.-q,sy-h+r);	
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		P2 = init_point2D(sx-5.*a/2.+q,sy-h+r);	
		if (i==N-1) P2.x -= 2.*q;
		vec_add_mult(&mean,&P2,1.);	
		add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		
	}
	if (condname!=NULL) Bound_o[2] = P2;
	
	sx -= 5.*a/2.;
	for (j=0;j<M;j++){		
		sy = (double)(2*(M-1-j)+1)*h;		
		P2 = init_point2D(sx-a/2.-r,sy);			
		if (j<M-1){
			vec_add_mult(&mean,&P2,1.);	
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
			P2 = init_point2D(sx-r,sy-h);	
			vec_add_mult(&mean,&P2,1.);	
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
		}
		else line_close_Poly(&Points,&poly_size,&P1,1,&eq_distr);
	}
	if (condname!=NULL) Bound_o[3] = P1;	
	
	int* Sizes = zero_int_list(2*N*M+1);
	point2D* Outer = (point2D*)malloc(2*N*M*sizeof(point2D));
	Sizes[0] = poly_size;
	
	double g = fac*h;
	double b = fac*a;
	
	k = 1;
	for (i=0;i<2*N;i++){
		for (j=0;j<M;j++){
			old = poly_size;
			center = init_point2D((double)a*(1+3*i)/2,(double)h*(1+2*j)+(i % 2)*h);
			P1 = init_point2D(center.x-b/2.,center.y-g);
			
			P2 = init_point2D(center.x+b/2.,center.y-g);
			add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);
			P2 = init_point2D(center.x+b,center.y);
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
			P2 = init_point2D(center.x+b/2.,center.y+g);
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
			P2 = init_point2D(center.x-b/2.,center.y+g);
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);
			P2 = init_point2D(center.x-b,center.y);
			add_line_to_Poly(&Points,&poly_size,NULL,&P2,1,&eq_distr);		
			line_close_Poly(&Points,&poly_size,&P1,1,&eq_distr);
	
			Outer[k-1] = init_point2D(center.x,center.y);
			Sizes[k] = poly_size-old;	
			
			if (condname!=NULL) Bound_i[k-1] = P1;	
			
			k++;				
		}
	}
		
	int* Loop_start = zero_int_list(2*N*M+2);
	int total_size = 0;
	for (k=1;k<2*N*M+2;k++){
		total_size += Sizes[k-1];
		Loop_start[k] = total_size;
	}
	
	//edge_refine(&Points,Loop_start,2*N*M+1,M_PI/6.,min_edge_len,edge_refine_fraction);
	
	point2D** Polys = (point2D**)malloc((2*N*M+1)*sizeof(point2D*));
	for (k=0;k<2*N*M+1;k++){
		Sizes[k] = Loop_start[k+1]-Loop_start[k];
		Polys[k] = (point2D*)malloc(Sizes[k]*sizeof(point2D));		
		for (j=Loop_start[k];j<Loop_start[k+1];j++) Polys[k][j-Loop_start[k]] = init_point2D(Points[j].x,Points[j].y);
	}

	create_nonconvex_poly_file(name,Polys,Sizes,2*N*M+1,Outer);
	
	/*if (condname!=NULL){
		int cn = 3;
		int* Cond = {0,1,5};
		create_condition_template(condname,Polys,hole_num+1,Cond,cn);
	}*/
	
	for (k=0;k<2*N*M+1;k++) free(Polys[k]);
	free(Polys);
	free(Loop_start);
	free(Points);
	free(Sizes);
	free(Outer);	
	
	FILE* Cond_file = fopen(condname,"w");
	if (Cond_file!=NULL){
		fprintf(Cond_file,"begin(BC)\n");
		
		fprintf(Cond_file,"var 0\n");		
		fprintf(Cond_file,"(%f,%f)\tROBIN(const,0)\t(%f,%f)\tNEUMANN(const,0)\t(%f,%f)\tROBIN(const,0)\t(%f,%f)\tNEUMANN(const,0)\n",
		 Bound_o[0].x,Bound_o[0].y,Bound_o[1].x,Bound_o[1].y,Bound_o[2].x,Bound_o[2].y,Bound_o[3].x,Bound_o[3].y);
		for (k=0;k<2*N*M;k++) fprintf(Cond_file,"(%f,%f)\tNEUMANN(const,0)\n",Bound_i[k].x,Bound_i[k].y);	
		
		fprintf(Cond_file,"var 1\n");		
		fprintf(Cond_file,"(%f,%f)\tROBIN(const,0)\t(%f,%f)\tNEUMANN(const,0)\t(%f,%f)\tROBIN(const,0)\t(%f,%f)\tNEUMANN(const,0)\n",
		 Bound_o[0].x,Bound_o[0].y,Bound_o[1].x,Bound_o[1].y,Bound_o[2].x,Bound_o[2].y,Bound_o[3].x,Bound_o[3].y);
		for (k=0;k<2*N*M;k++) fprintf(Cond_file,"(%f,%f)\tNEUMANN(const,0)\n",Bound_i[k].x,Bound_i[k].y);	
		
		fprintf(Cond_file,"var 5\n");		
		fprintf(Cond_file,"(%f,%f)\tDIRICHLET(const,0)\n",Bound_o[0].x,Bound_o[0].y);
		for (k=0;k<2*N*M;k++) fprintf(Cond_file,"(%f,%f)\tDIRICHLET(linear,293,293)\n",Bound_i[k].x,Bound_i[k].y);	
		
		fprintf(Cond_file,"end(BC)\n\n");

		fprintf(Cond_file,"begin(IC)\n");
		fprintf(Cond_file,"var 0 const=0.0\n");
		fprintf(Cond_file,"var 1 const=0.0\n");
		fprintf(Cond_file,"var 2 const=1.0\n");
		fprintf(Cond_file,"var 3 function=\n");
		fprintf(Cond_file,"var 4 const=0.0\n");
		fprintf(Cond_file,"var 5 const=293.0\n");
		
		fprintf(Cond_file,"end(IC)\n");
		
		free(Bound_o);
		free(Bound_i);
		
		fclose(Cond_file);
	}
	
	return (double)3*a*h*N*M/mesh_size;
}

double create_shape9(char* name,int mesh_size){
	
	const double r = 0.35;
	const double R = 0.5;
	const double d = 0.075;

	const int hole_num = 4;
	const int m = 200;
	const int q = 100;
	const int s = 50;
	
	int i,j,k,old;
	double x,y;
	point2D P1,P2;
	
	int poly_size = 0;
	point2D* Points = NULL;
	point2D* Outer = (point2D*)malloc(hole_num*sizeof(point2D));
	int* Sizes = (int*)malloc((hole_num+1)*sizeof(int));
	
	double angle = M_PI/2.-2.*asin(d/(2.*r));
	point2D mean = {.x=r/sqrt(8.)+d/4.,.y=r/sqrt(8.)+d/4.};
	point2D center = {.x=0,.y=R};
		
	add_circle_to_Poly(&Points,&poly_size,&center,R,m);
	Sizes[0] = poly_size;
	
	old = poly_size;
	P1 = init_point2D(d/2.,d/2.+R);
	P2 = init_point2D(sqrt(r*r-d*d/4.),d/2.+R);
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);	
	add_arc_to_Poly(&Points,&poly_size,NULL,&center,angle,m/4,&eq_distr);
	line_close_Poly(&Points,&poly_size,&P1,1,eq_distr);
	Outer[0] = init_point2D(mean.x,mean.y+R);
	Sizes[1] = poly_size-old;
	
	old = poly_size;
	P1 = init_point2D(-d/2.,d/2.+R);
	P2 = init_point2D(-d/2,sqrt(r*r-d*d/4.)+R);
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);												
	add_arc_to_Poly(&Points,&poly_size,NULL,&center,angle,m/4,&eq_distr);												
	line_close_Poly(&Points,&poly_size,&P1,1,eq_distr);												
	Outer[1] = init_point2D(-mean.x,mean.y+R);
	Sizes[2] = poly_size-old;									
	
	old = poly_size;										
	P1 = init_point2D(-d/2.,-d/2.+R);
	P2 = init_point2D(-sqrt(r*r-d*d/4.),-d/2+R);										
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);												
	add_arc_to_Poly(&Points,&poly_size,NULL,&center,angle,m/4,&eq_distr);												
	line_close_Poly(&Points,&poly_size,&P1,1,eq_distr);												
	Outer[2] = init_point2D(-mean.x,-mean.y+R);
	Sizes[3] = poly_size-old;										
	
	old = poly_size;										
	P1 = init_point2D(d/2.,-d/2.+R);
	P2 = init_point2D(d/2.,-sqrt(r*r-d*d/4.)+R);								
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,1,&eq_distr);												
	add_arc_to_Poly(&Points,&poly_size,NULL,&center,angle,m/4,&eq_distr);												
	line_close_Poly(&Points,&poly_size,&P1,1,eq_distr);												
	Outer[3] = init_point2D(mean.x,-mean.y+R);
	Sizes[4] = poly_size-old;									
									

	int total_size = 0;
	int* Loop_start = zero_int_list(hole_num+2);
	for (k=1;k<hole_num+2;k++){
		total_size += Sizes[k-1];
		Loop_start[k] = total_size;
	}
	edge_refine(&Points,Loop_start,hole_num+1,M_PI/6.,min_edge_len,edge_refine_fraction);
	
	point2D** Polys = (point2D**)malloc((hole_num+1)*sizeof(point2D*));
	for (k=0;k<hole_num+1;k++){
		Sizes[k] = Loop_start[k+1]-Loop_start[k];
		Polys[k] = (point2D*)malloc(Sizes[k]*sizeof(point2D));		
		for (j=Loop_start[k];j<Loop_start[k+1];j++) Polys[k][j-Loop_start[k]] = init_point2D(Points[j].x,Points[j].y);
	}
	
	create_nonconvex_poly_file(name,Polys,Sizes,hole_num+1,Outer);
	
	for (k=0;k<hole_num+1;k++) free(Polys[k]);
	free(Polys);
	free(Loop_start);
	free(Points);
	free(Sizes);
	free(Outer);
	return (double)8.*R*R/(1.3*mesh_size);
}

 // Attention: point density at attribute bounds must be larger than refinement, otherwise spred values will fill all the domains !

double create_attr1(char* name,int mesh_size){
	const int n = 100;
	const int m = 100;
	const double l = 1.;
	const double aspectratio = 1.;
	const int rows = 1;
	const int cols = 1;
	const double a = 0.5;
	const double b = 0.5;
	
	int i,j,k,old;
	double s;
	point2D P1,P2;
	
	int poly_size = 0;
	point2D* Points = NULL;
	P1 = init_point2D(-l/(2.*aspectratio),0);										//	 ________
	P2 = init_point2D(-l/(2.*aspectratio),l);										//	|    _   |
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,n,&eq_distr);						//	|	|_|  |
	P2 = init_point2D(l/(2.*aspectratio),l);										//  |________|
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,n,&eq_distr);
	P2 = init_point2D(l/(2.*aspectratio),0);
	add_line_to_Poly(&Points,&poly_size,NULL,&P2,n,&eq_distr);
	line_close_Poly(&Points,&poly_size,&P1,n,&eq_distr);
	
	int* Sizes = zero_int_list(rows*cols+1);
	double* Attr = zero_vector(rows*cols+1);
	point2D* Outer = (point2D*)malloc((rows*cols+1)*sizeof(point2D));
	Sizes[0] = poly_size;
	Outer[0] = init_point2D(0,0.5);
	Attr[0] = 1.0;
	for (i=0;i<rows;i++){
		for (j=0;j<cols;j++){
			old = poly_size;
			point2D br = init_point2D(-l/(2.*aspectratio)+(double)(j+1)*l/(aspectratio*(cols+1))-a/2.,(double)(i+1)*l/(rows+1)-b/2.);			
			P1 = init_point2D(br.x+a,br.y);
			add_line_to_Poly(&Points,&poly_size,&br,&P1,m,&eq_distr);
			P1 = init_point2D(br.x+a,br.y+b);
			add_line_to_Poly(&Points,&poly_size,NULL,&P1,m,&eq_distr);
			P1 = init_point2D(br.x,br.y+b);
			add_line_to_Poly(&Points,&poly_size,NULL,&P1,m,&eq_distr);
			line_close_Poly(&Points,&poly_size,&br,m,&eq_distr);
			Outer[i*cols+j+1] = init_point2D(br.x+a/2.,br.y+b/2.);
			Sizes[i*cols+j+1] = poly_size-old;
			Attr[i*cols+j+1] = -1.0;
		}
	}
	
	int total_size = 0;
	int* Loop_start = zero_int_list(rows*cols+2);
	for (k=1;k<rows*cols+2;k++){
		total_size += Sizes[k-1];
		Loop_start[k] = total_size;
	}
	
	//edge_refine(&Points,Loop_start,rows*cols+1,M_PI/6.,min_edge_len,edge_refine_fraction);
	
	point2D** Polys = (point2D**)malloc((rows*cols+1)*sizeof(point2D*));
	for (k=0;k<rows*cols+1;k++){
		Sizes[k] = Loop_start[k+1]-Loop_start[k];
		Polys[k] = (point2D*)malloc(Sizes[k]*sizeof(point2D));		
		for (j=Loop_start[k];j<Loop_start[k+1];j++) Polys[k][j-Loop_start[k]] = init_point2D(Points[j].x,Points[j].y);
	}

	create_attribute_poly_file(name,Polys,Sizes,rows*cols+1,Outer,Attr);
	
	for (k=0;k<rows*cols+1;k++) free(Polys[k]);
	free(Polys);
	free(Loop_start);
	free(Points);
	free(Sizes);
	free(Outer);
	free(Attr);
	
	return (double)2.*l*l/(1.3*mesh_size);
}

double create_attr2(char* name,int mesh_size){
	const int nx = 1;
	const int ny = 10;
	const int regions = 4;
	const double l = 1.;
	const double aspectratio = 1.;
	const double r = 0.1;
	const double d = 0.02;
	const double f = 0.2;
	const double s = 0.05;
	const double ms = 0.2;
	
	int i;
	point2D P1,P2;
	
	int m = 600;//(int)round((double)2.*M_PI*r*n/l);
	int q = 20;//(int)round((double)M_PI*d*n/l);
	int p = 1000;
	
	int* Sizes = zero_int_list(regions);
	int* Start = zero_int_list(regions);
	double* Attr = zero_vector(regions);
	point2D* Outer = (point2D*)malloc(regions*sizeof(point2D));
	
	int poly_size = 0;
	point2D* Points = NULL;
	Start[0] = 0;
	
	P1 = init_point2D(-l/(2.*aspectratio),0);	
	P2 = init_point2D(-l/(2.*aspectratio),l);						
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,ny,&eq_distr);
	P1 = init_point2D(-d/2.,l);									
	add_line_to_Poly(&Points,&poly_size,NULL,&P1,nx,&eq_distr);
	
	P1 = init_point2D(-d/2.,l-f);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P1,nx,&eq_distr);		
	P2 = init_point2D(0,l-f);
	add_arc_to_Poly(&Points,&poly_size,&P1,&P2,M_PI,q,&eq_distr);
	
	P1 = init_point2D(d/2.,l);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P1,nx,&eq_distr);			
	P1 = init_point2D(l/(2.*aspectratio),l);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P1,nx,&eq_distr);			
	P1 = init_point2D(l/(2.*aspectratio),0);	
	add_line_to_Poly(&Points,&poly_size,NULL,&P1,ny,&eq_distr);	
	
	Sizes[0] = poly_size;
	Outer[0] = init_point2D(0,l/10.);
	Attr[0] = 1.0;
	Start[1] = poly_size;
	
	P1 = init_point2D(s,l/2.);
	add_circle_to_Poly(&Points,&poly_size,&P1,r,m);		
	
	Sizes[1] = poly_size-Start[1];
	Outer[1] = init_point2D(s,l/2.);
	Attr[1] = -1.0;
	Start[2] = poly_size;
	
	P1 = init_point2D(-l/(2.*aspectratio)+l*ms,l);
	P2 = init_point2D(-l/(2.*aspectratio)+l*ms,0);
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,p,&eq_distr);
	Sizes[2] = poly_size-Start[2];
	Outer[2] = init_point2D(-l/(2.*aspectratio)+l*ms/2.,l/2.);
	Attr[2] = -1.0;
	Start[3] = poly_size;
	
	P1 = init_point2D(l/(2.*aspectratio)-l*ms,l);
	P2 = init_point2D(l/(2.*aspectratio)-l*ms,0);
	add_line_to_Poly(&Points,&poly_size,&P1,&P2,p,&eq_distr);
	Sizes[3] = poly_size-Start[3];
	Outer[3] = init_point2D(+l/(2.*aspectratio)-l*ms/2.,l/2.);
	Attr[3] = -1.0;
	
	point2D** Polys = (point2D**)malloc(2*sizeof(point2D*));
	for (i=0;i<regions;i++) Polys[i] = &(Points[Start[i]]);
	create_attribute_poly_file(name,Polys,Sizes,regions,Outer,Attr);
	
	return (double)2.*l*l/(1.3*mesh_size);
}



/////////////////////// sim functions ///////////////////////////////////////////////////////////////////

void init_2D_eigen_field(eigen_field** Field,double* Vec1,double* Vec2,double* Eig){
	int i;
	double a,b,nx,ny;
	
	int n = glob_mesh.size;
	*Field = (eigen_field*)malloc(sizeof(eigen_field));
	(*Field)->dim = 2;
	(*Field)->field_size = n;
	
	(*Field)->Eig = (double**)malloc(((*Field)->dim)*sizeof(double*));
	(*Field)->Eig[0] = (double*)malloc(n*sizeof(double));
	(*Field)->Eig[1] = (double*)malloc(n*sizeof(double));

	(*Field)->Vec = (double**)malloc(((*Field)->dim)*sizeof(double*));
	(*Field)->Vec[0] = (double*)malloc(((*Field)->dim)*n*sizeof(double));
	(*Field)->Vec[1] = (double*)malloc(((*Field)->dim)*n*sizeof(double));
	
	for (i=0;i<n;i++){
		if (Eig[i]<Eig[n+i]){
			(*Field)->Eig[0][i] = Eig[i];
			(*Field)->Eig[1][i] = Eig[n+i];
			(*Field)->Vec[0][i] = Vec1[i];
			(*Field)->Vec[0][n+i] = Vec1[n+i];
			(*Field)->Vec[1][i] = Vec2[i];
			(*Field)->Vec[1][n+i] = Vec2[n+i];
		}
		else{
			(*Field)->Eig[0][i] = Eig[n+i];
			(*Field)->Eig[1][i] = Eig[i];
			(*Field)->Vec[0][i] = Vec2[i];
			(*Field)->Vec[0][n+i] = Vec2[n+i];
			(*Field)->Vec[1][i] = Vec1[i];
			(*Field)->Vec[1][n+i] = Vec1[n+i];
		}
	}
	
}

void free_eigen_field(eigen_field** Field){
	int i;
	if ((*Field)==NULL) return;
	if ((*Field)->Eig!=NULL){
		for (i=0;i<(*Field)->dim;i++) free((*Field)->Eig[i]);
		free((*Field)->Eig);		
	}
	if ((*Field)->Vec!=NULL){
		for (i=0;i<(*Field)->dim;i++) free((*Field)->Vec[i]);
		free((*Field)->Vec);
	}
	free(*Field);
	*Field = NULL;	
}

int load_bin_image_data(char* FullName,int*** Data,int* Nx,int* Ny){
	const int x_estimate = 10000;

	int i,j,k;
	char c;
	
	FILE* file = fopen(FullName,"r");
	if (file!=NULL){
		*Nx = x_estimate;
		*Ny = 1;
		*Data = (int**)malloc(sizeof(int*));
		(*Data)[0] = (int*)malloc((*Nx)*sizeof(int));
		i = 0;
		j = 0;
		do{
			k = fgetc(file);
			if (k!=EOF){
				c = (char)k;
				switch(c){
					case '0': (*Data)[j][i] = 0;break;
					case '1': (*Data)[j][i] = 1;break;
					case '\t': i++;break;
					case '\n':
						if (j==0){
							*Nx = i+1;
							(*Data)[j] = (int*)realloc((*Data)[j],(*Nx)*sizeof(int));
						}
						j++;
						*Data = (int**)realloc(*Data,(j+1)*sizeof(int*));
						(*Data)[j] = (int*)malloc((*Nx)*sizeof(int));
						i = 0;
						break;
					default: printf("unknown character encountered");
				}
			}
		}while(k!=EOF);
		*Ny = j+1;
		fclose(file);
		return 1;
	}
	else{
		printf("could not open file %s\n",FullName);	
		perror(FullName);		
		return 0;
	}
}

index2D image_map(double x,double y,int nx,int ny){
	const double scale = 1.05;
	index2D res;
	
	res.i = (int)floor((double)(x/scale+0.5)*nx);
	res.j = (int)floor((double)((y-0.5)/scale+0.5)*ny);

	if (res.i<0) res.i = 0;	
	if (res.i>=nx) res.i = nx-1;
	
	if (res.j<0) res.j = 0;	
	if (res.j>=nx) res.j = ny-1;
	
	res.j = ny-res.j-1;
	
	return res;
}

double create_shape_from_image(char* name,int mesh_size){
	
	char* DataName = "/Home/damage/radszuwe/Desktop/radszuwe/Latex-Dokumente/GAMM2015/figures/data.dat";
	//char* MapName = "/Home/damage/radszuwe/Desktop/radszuwe/Latex-Dokumente/GAMM2015/figures/data.map";
	const double out = 1e20;
	const double scale = 1.;
	
	int i,j;
	double x,y;
	
	int nx = 0;
	int ny = 0;
	double l = scale;
	int** Data = NULL;
	point2D topleft = init_point2D(out,-out);
	point2D topright = init_point2D(-out,-out);
	point2D bottomleft = init_point2D(out,out);
	point2D bottomright = init_point2D(-out,out);
	
	if (load_bin_image_data(DataName,&Data,&nx,&ny)==SUCCESS){
		for (j=0;j<ny;j++){
			for (i=0;i<nx;i++){
				x = (double)scale*(i-nx/2.)/nx;
				y = (double)scale*j/ny;
				if (Data[i][j]==0){
					if (x<topleft.x) topleft = init_point2D(x,y);				
					if (y>topright.y) topright = init_point2D(x,y);	
					if (y<bottomleft.y) bottomleft = init_point2D(x,y);	
					if (x>bottomright.x) bottomright = init_point2D(x,y);	
				}
			}
		}
		
		int poly_size = 0;
		point2D* Points = NULL;								
		add_line_to_Poly(&Points,&poly_size,&topleft,&topright,1,&eq_distr);	
		add_line_to_Poly(&Points,&poly_size,NULL,&bottomright,1,&eq_distr);
		add_line_to_Poly(&Points,&poly_size,NULL,&bottomleft,1,&eq_distr);
		add_line_to_Poly(&Points,&poly_size,NULL,&topleft,1,&eq_distr);
		
		int* Loop_start = zero_int_list(2);
		Loop_start[1] = poly_size;
		edge_refine(&Points,Loop_start,1,M_PI/6.,min_edge_len,edge_refine_fraction);
		poly_size = Loop_start[1];
		free(Loop_start);
	
		create_poly_file(name,Points,poly_size);
		return (double)2.*l*l/(1.3*mesh_size);		
	}
	else{
		printf("could not create mesh from %s -> abort\n",name);
		exit(0);
		return 0;
	}
}

double initial_conditions_from_image(double x,double y,char* DataName){
	static int** Data = NULL;
	static int nx = 0;
	static int ny = 0;
	
	int i,j;
	
	if (strcmp(DataName,"clean")==0){
		for (j=0;j<ny;j++) free(Data[j]);
		free(Data);
		return 0;
	}
	else if (strcmp(DataName,"compute")==0){
		index2D ind = image_map(x,y,nx,ny);
		//return (double)2.*Data[ind.j][ind.i]-1.;	// [-1,1]
		return (double)(-2.*Data[ind.j][ind.i]+1.);	// change values
		//return (double)Data[ind.j][ind.i];    // [0,1]
	}
	else if (DataName!=NULL){
		load_bin_image_data(DataName,&Data,&nx,&ny);
		return 0;
	}
	return 0;
}

double initial_conditions_from_image_compute(double x,double y){
	return initial_conditions_from_image(x,y,"compute");
}

double* Bend_U(double b){
	int i;
	double x,y;
	int n = glob_mesh.size;
	double* Res = zero_vector(2*n);
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		Res[i] = x*sin(b)*(y-0.5);
		Res[n+i] = (cos(b)-1.)*(y-0.5);
	}
	return Res;
}

int get_base_poly_size(char* BaseName){
	char BasePolyName[512];
	char Buffer[512];
	sprintf(BasePolyName,"%s/%s.poly",Mesh_dir,BaseName);
	FILE* file = fopen(BasePolyName,"r");
	if (file==NULL){
		printf("could not open file %s -> abort\n",BasePolyName);
		exit(0);
	}
	int res = 0;
	if (readline(file,Buffer)){
		int i,len;
		char** Parts = split(Buffer," ",&len);
		if (len>0) res = atoi(Parts[0]);
		for (i=0;i<len;i++) free(Parts[i]);
		free(Parts);
	}
	fclose(file);
	return res;
}

void create_refined_mesh(double* Refinement_data,double* H,double* C,int quiet){
	
	int i,j,n_base;	
	char** Parts;
	char BasePolyName[512];
	char BaseName[512];
	char PolyName[512];
	char FullName[512];
	char FullDir[512];
	char Command[1024];	
	char Buffer[1024];	

	int size = 0;
	int* Nodes = NULL;		
	int len = 0;
	double max_old = max_triangle_size(&glob_mesh,&elements);
	Parts = split(Mesh_name,".",&len);
	strcpy(BaseName,Parts[0]);
	for (i=0;i<len;i++) free(Parts[i]);
	free(Parts);
	glob_mesh_index++;
	
	get_crack_set(H,crack_thres,&Nodes,&size);
	
	#ifdef CONSTANT_PHASE 
		if (C!=NULL){
			int m = glob_mesh.size;
			int* List = zero_int_list(m);
			double* G = zero_vector(m);
			double* Refine = zero_vector(m);
			refine_by_gradient(C,Refine,1,min_refine_area,max_refine_area);
			j = 0;
			for (i=0;i<m;i++) if (Refine[i]<=10.*min_refine_area){
				List[j] = i;
				j++;
			}
			List = (int*)realloc(List,j*sizeof(int));
		
			int new_size = 0;
			int* Temp = clone_list(Nodes,size);
			if (Nodes!=NULL) free(Nodes); Nodes = NULL;
			join_lists(&Nodes,&new_size,Temp,List,size,j);
			size = new_size;
		
			free(G);
			free(Refine);
			free(Temp);		
		}
	#endif
	
	if (Crack_constrained==YES && size>0){
		
		// obtain set of nodes of crack
		n_base = get_base_poly_size(BaseName);
		point2D* CrackSet = (point2D*)malloc(size*sizeof(point2D));
		i = 0;
		for (j=0;j<size;j++) if (Nodes[j]>=n_base && is_boundary(Nodes[j])<0){
			CrackSet[i] = init_point2D(glob_mesh.Points[Nodes[j]].x,glob_mesh.Points[Nodes[j]].y);
			i++;
		}
		size = i;
		CrackSet = (point2D*)realloc(CrackSet,size*sizeof(point2D));
		//print_point_list2D(CrackSet,size,"points");
		
		
		// create new poly file with crack set as constraints
		sprintf(BasePolyName,"%s/%s.poly",Mesh_dir,BaseName);
		sprintf(PolyName,"%s/%s/temp.poly",Output_dir,Output_name);
		add_constraint_points(BasePolyName,PolyName,CrackSet,size);
		
		// obtain area information from base mesh
		double area = -1.;
		sprintf(FullName,"%d",getpid());
		sprintf(Command,"grep -e \"triangle\" %s/%s.node > %s",Mesh_dir,Mesh_name,FullName);
		system(Command);
		FILE* Info = fopen(FullName,"r");
		readline(Info,Buffer);
		fclose(Info);
		sprintf(Command,"rm %s",FullName);
		system(Command);
		Parts = split(Buffer," ",&len);
		for (i=0;i<len;i++) if (strstr(Parts[i],"-a")!=NULL){
			int sublen;
			char** SubParts = split(Parts[i],"a",&sublen);
			if (sublen==2) area = atof(SubParts[1]);
			for (j=0;j<sublen;j++) free(SubParts[j]);
			free(SubParts);
			break;
		}
		for (i=0;i<len;i++) free(Parts[i]);
		free(Parts);
		if (area<0){			
			printf("create_refined_mesh: no area info found in file %s/%s -> use standard value\n",Output_dir,Mesh_name);
			area = (double)get_total_area()/glob_mesh.size;
		}
		
		// create mesh from poly file
		sprintf(Command,"triangle -p -F -Q -q30 -a%f %s",area,PolyName);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
		
		// interpolate refinement data 	
		mesh2D temp_mesh;
		element_collection temp_elements;
		sprintf(FullDir,"%s/%s",Output_dir,Output_name);
		read_mesh_2D(FullDir,"temp.1",&temp_mesh,&temp_elements,NULL,NULL,QUIET);			
		double* Int_refine_data = get_minimum_interpolation(Refinement_data,&temp_mesh,&temp_elements,&glob_mesh,&elements);
		
		// refine and copy to mesh folder
		sprintf(FullName,"%s/%s/temp.1",Output_dir,Output_name);
		Refine_mesh(FullName,Int_refine_data,temp_elements.size,quiet);
		free(Int_refine_data);
		sprintf(Command,"mv %s/%s/temp.2.node %s/%s/meshes/%s.%d.node",Output_dir,Output_name,Output_dir,Output_name,BaseName,glob_mesh_index);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
		sprintf(Command,"mv %s/%s/temp.2.ele %s/%s/meshes/%s.%d.ele",Output_dir,Output_name,Output_dir,Output_name,BaseName,glob_mesh_index);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
		
		// clean
		FreeMesh(&temp_mesh);
		free(temp_elements.Elements);
		free(CrackSet);
	}
	else{		
		// interpolate point data of current mesh to triangle data of base mesh
				
		double* Int_refine_data = get_minimum_interpolation(Refinement_data,&base_mesh,&base_elements,&glob_mesh,&elements);
		
		// create .area file and let triangle create refined mesh
		sprintf(FullName,"%s/%s/%s.1",Output_dir,Output_name,BaseName);	
		Refine_mesh(FullName,Int_refine_data,base_elements.size,quiet);
		free(Int_refine_data);
		
		// copy to proper dir
		sprintf(Command,"mv %s/%s/%s.2.node %s/%s/meshes/%s.%d.node",Output_dir,Output_name,BaseName,Output_dir,Output_name,BaseName,glob_mesh_index);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
	
		sprintf(Command,"mv %s/%s/%s.2.ele %s/%s/meshes/%s.%d.ele",Output_dir,Output_name,BaseName,Output_dir,Output_name,BaseName,glob_mesh_index);
		if (!quiet) printf("execute: %s\n",Command);
		system(Command);
	}
	
	if (Nodes!=NULL) free(Nodes);
}

// functions simulation mode user defined /////////////////////////////////////////////////////////////////////

double indentation(double x,double y,double t,int var_index){
	const double pos = 0.1;
	const double r = 0.1;
	const double dmax = 0.05;
	if (var_index==0) return 0;
	
	if (y>0.5 && x>=pos-r && x<= pos+r){
		return -cos(M_PI/2.*(x-pos)/r)*dmax*t;
	}
	else return 0;
}

double U_bound_val(int node,int var_index,double t){					
	double res,x,y,b,s;
	double b_min = 0.0;
	//double b_max = M_PI/8.; 												// deformation max 15% for bending
	double b_max = 1.0;//0.064;														// for stretch/shear
	
	x = glob_mesh.Points[node].x;
	y = glob_mesh.Points[node].y;
	//s = t/total_time<0.7 ? t/total_time : 0.7;							// saturation
	//s = sin(4*M_PI*t/total_time);											// periodic in time
	s = t/total_time;														// linear increase
	
	
	b = linear(t,total_time,b_min,b_max);							
	//b = linear_up_down(t,total_time*0.16,b_min,b_max);
	switch(var_index){
		case 0: 			
			//res = periodic_stretch_shear(x,y,s,var_index);
			//res = (x>=0 ? b : -b);										// two sided x stretch
			//res = (x>0 ? b : 0);											// right sided x stretch	
			res = indentation(x,y,s,var_index);
			//res = (x>0 ? 2.*b*(y-0.5) : -2.*b*(y-0.5));					// bending linear
			//res = (x>0 ? sin(b)*(y-0.5) : -sin(b)*(y-0.5));				// bending 
			break;
		case 1:
			//res = periodic_stretch_shear(x,y,s,var_index);		
			res = indentation(x,y,s,var_index);
			//res = 0;
			break;
		default: res = 0;												
	}
	return res;
}

double U_bound_force(int node,int var_index,double t){
	const double tol = 1e-10;
	const double f_min = 0;
	const double f_max = 10.;
	const double l = 0.5;
	
	double res,x,y,f;
	x = glob_mesh.Points[node].x;
	y = glob_mesh.Points[node].y;
	
	f = f_min+(f_max-f_min)*t/total_time;									// linear increase
	
	switch(var_index){
		case 0:
			//res = (fabs(x-l)<tol ? f: 0);
			res = 0;
			break;
		case 1:			
			res = 0;
			/*if (fabs(y-l)<tol) res = f;
			else if (fabs(y)<tol) res = -f;
			else res = 0;*/
			break;			
	}
	return res;
}

bound_cond* get_U_bound_conditions(double t){
	const double tol = 1e-10;
	const double lx = 0.5;
	const double ly = 1.;
	
	int i,counter;
	double x,y;
	
	int d = glob_mesh.size;
	//double s = t;//total_time;				
	bound_cond* Res = (bound_cond*)malloc(sizeof(bound_cond));
	Res->size = dimension*d;
	Res->Cond = zero_int_list(dimension*d);
	Res->Val = zero_vector(dimension*d);
	
	if (strcmp(Cond_name,"none")==0){
		double b_min = 0.;
		double b_max = 1.;
		double b = linear(t,total_time,b_min,b_max);	
				
		counter = 0;
		for (i=0;i<d;i++){
			if (is_boundary(i)>=0){
				x = glob_mesh.Points[i].x;
				y = glob_mesh.Points[i].y;
			
				#ifdef X_STRETCH
					if (fabs(fabs(x)-lx)<tol){
						Res->Cond[i] = DIRICHLET;	
						Res->Val[i] = (x>0 ? b : 0);
						Res->Val[d+i] = 0.;								
					}
					else{
						Res->Cond[i] = NEUMANN;	
						Res->Val[i] = 0.;
						Res->Val[d+i] = 0.;					
					}
				#endif		
				
				#ifdef Y_STRETCH
					if(fabs(y)==0 || fabs(y-ly)==0){
						 Res->Cond[i] = DIRICHLET; 
						 Res->Val[i] = 0.;
						 Res->Val[d+i] = (y>ly/2. ? b :0);		
					 }
					 else{
						 Res->Cond[i] = NEUMANN;	 
						 Res->Val[i] = 0.;
						 Res->Val[d+i] = 0.;		
					 }
				#endif		
			
				#ifdef X_SHEAR
					if (fabs(fabs(x)-lx)<tol){
						Res->Cond[i] = DIRICHLET;
						Res->Val[i] = 0.;
						Res->Val[d+i] = b*x;
					}
					else{
						 Res->Cond[i] = NEUMANN;	 
						 Res->Val[i] = 0.;
						 Res->Val[d+i] = 0.;		
					 }
				#endif
			
				
				#ifdef Y_SHEAR
					if(fabs(y)==0 || fabs(y-ly)==0){
						Res->Cond[i] = DIRICHLET;
						Res->Val[i] = b*(y-ly/2.);
						Res->Val[d+i] = 0.;						
					}
					else{
						 Res->Cond[i] = NEUMANN;	 
						 Res->Val[i] = 0.;
						 Res->Val[d+i] = 0.;		
					 }
				#endif
			   
				#ifdef OTHER
					if (fabs(x+lx)<tol){
						Res->Cond[i] = DIRICHLET;
						Res->Val[i] = U_bound_val(i,0,t);
						Res->Val[d+i] = U_bound_val(i,1,t);		 	
					}
					else{
						 Res->Cond[i] = NEUMANN;	 
						 Res->Val[i] = U_bound_force(i,0,t);
						 Res->Val[d+i] = U_bound_force(i,1,t);		
					}
				#endif
			}
			else Res->Cond[i] = NOCON;
		
			if (Res->Cond[i]==DIRICHLET) counter++;
			if (counter>dimension-1) pure_neumann_reg_factor = 0;
			Res->Cond[d+i] = Res->Cond[i];
		}
	}
	else{		
		char* Expr;
		bound_info* Info_x = Boundary_conditions[0];								// ux and uy must have same BC type
		bound_info* Info_y = Boundary_conditions[1];
		if (Info_x!=NULL && Info_y!=NULL){
			int* GeomConds = get_boundary_conditions(Info_x->Coords,Info_x->Cond,Info_x->Sizes,Info_x->loops);
			for (i=0;i<d;i++) if (GeomConds[i]!=NOCON){
				x = glob_mesh.Points[i].x;
				y = glob_mesh.Points[i].y;
				Res->Cond[i] = GeomConds[i];
				Res->Cond[d+i] = GeomConds[i];
				switch(GeomConds[i]){
					default:
						Expr = Info_x->Time_depend[GeomConds[d+i]][GeomConds[2*d+i]];
						Res->Val[i] = parse_bound_expr(Expr,x,y,t);
						Expr = Info_y->Time_depend[GeomConds[d+i]][GeomConds[2*d+i]];
						Res->Val[d+i] = parse_bound_expr(Expr,x,y,t);					
						break;
				}
			}
			free(GeomConds);
		}
		else{
			printf("no valid boundary conditons found for displacement field -> abort\n");
			exit(0);
		}
	}
	return Res;
}

int* default_segment_conditions(int var){

	int i;
	double x,y;
	
	int n = glob_mesh.size;
	
	int* Res = zero_int_list(n);
	for (i=0;i<n;i++) if (is_boundary(i)>=0){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		
		#ifdef X_STRETCH
			#define BOUND x>0
		#endif
		#ifdef X_SHEAR
			#define BOUND y>0.5
		#endif
		#ifdef Y_STRETCH
			#define BOUND y>0.5
		#endif
		#ifdef Y_SHEAR
			#define BOUND x>0
		#endif
		#ifndef BOUND
			#define BOUND 0
		#endif
		
		Res[i] = (BOUND) ? DIRICHLET : NOCON;

		#undef BOUND
	}
	else Res[i] = NOCON;		
	
	return Res;
}

int* get_segment_conditions(int var,int loop,int segment){
	if (strcmp(Cond_name,"none")==0) return default_segment_conditions(var);
	else{
		int i,l,s;
		int d = glob_mesh.size;
		bound_info* Info = Boundary_conditions[0];
		if (Info!=NULL){
			int* Res = zero_int_list(d);
			int* GeomConds = get_boundary_conditions(Info->Coords,Info->Cond,Info->Sizes,Info->loops);
			for (i=0;i<d;i++) if (GeomConds[i]!=NOCON){
				l = GeomConds[d+i];
				s = GeomConds[2*d+i];
				if (l==loop && s==segment) Res[i] = GeomConds[i]; else Res[i] = NOCON;
			}
			else Res[i] = NOCON;
			
			free(GeomConds);
			return Res;
		}
		else return NULL;
	}
}

double T_bound_val(int node,double t){
	double x,y;
	x = glob_mesh.Points[node].x;
	y = glob_mesh.Points[node].y;
	
	return 0;
}

double T_bound_flux(int node,double t,double local_T,double outer_T){	                                                       
	//double x,y;
	double outer_abs = outer_T+T_absolute_ref;
	double local_abs = local_T+T_absolute_ref;
	//x = glob_mesh.Points[node].x;
	//y = glob_mesh.Points[node].y;
	return kappa_bound*(outer_T-local_T)+surface_emissivity*Boltz*(outer_abs*outer_abs*outer_abs*outer_abs-local_abs*local_abs*local_abs*local_abs);
}

bound_cond* get_T_bound_conditions(double* T,double t){
	const double tol = 1e-10;
	const double lx = 0.5;
	const double ly = 1.;
	
	int i;
	double x,y;
	
	//double s = t/total_time;	
	int d = glob_mesh.size;
	bound_cond* Res = (bound_cond*)malloc(sizeof(bound_cond));
	Res->size = d;
	Res->Cond = zero_int_list(d);
	Res->Val = zero_vector(d);
	
	if (strcmp(Cond_name,"none")==0){
		
		for (i=0;i<d;i++){
			if (is_boundary(i)>=0){
				x = glob_mesh.Points[i].x;
				y = glob_mesh.Points[i].y;
								
				// Robin
				Res->Cond[i] = ROBIN;
				Res->Val[i] = T_bound_flux(i,t,T[i],T_outer);
				
				// Neumann
				//Res->Cond[i] = NEUMANN;
				//Res->Val[i] = 0;
				
				// Dirichlet
				//Res->Cond[i] = DIRICHLET;
				//Res->Val[i] = T_bound_val(i,t);
				
				//if (fabs(y-ly)<tol || fabs(y)<tol || fabs(fabs(x)-lx)<tol) Res[i] = DIRICHLET; else Res[i] = NEUMANN;	// shape 1,2,3,4
				//if (fabs(y-3.*ly/2.)<tol || fabs(y)<tol) Res[i] = DIRICHLET; else Res[i] = NEUMANN; // shape 5
				
			}
			else Res->Cond[i] = NOCON;
		}
	}
	else{
		bound_info* Info = Boundary_conditions[5];
		if (Info!=NULL){
			double f;
			char* Expr;
			int* GeomConds = get_boundary_conditions(Info->Coords,Info->Cond,Info->Sizes,Info->loops);
			for (i=0;i<d;i++) if (GeomConds[i]!=NOCON){
				x = glob_mesh.Points[i].x;
				y = glob_mesh.Points[i].y;
				Res->Cond[i] = GeomConds[i];				
				Expr = Info->Time_depend[GeomConds[d+i]][GeomConds[2*d+i]];
				switch(GeomConds[i]){					
					case DIRICHLET:							
						Res->Val[i] = parse_bound_expr(Expr,x,y,t);													
						break;
					case NEUMANN: 
						Res->Val[i] = parse_bound_expr(Expr,x,y,t);													
						break;
					case ROBIN:
						f = parse_bound_expr(Expr,x,y,t);										
						Res->Val[i] = T_bound_flux(i,t,T[i],f);			
						break;
					default:
						Res->Val[i] = 0;
				}
				
			}
			free(GeomConds);			
		}
		else{
			printf("no valid boundary conditons found for temperature field -> abort\n");
			exit(0);
		}		
	}
	return Res;
}

void force_constraints(double* H,double min,double max){
	int i;
	int n = glob_mesh.size;
	
	for (i=0;i<n;i++){
		if (H[i]<min) H[i] = min;
		if (H[i]>max) H[i] = max;
	}
}

void set_initial_conditions(double* X){											
	const double tol = 1e-10;
	const double l = 1.;
	
	static char Expression[MAX_FUNC_SIZE];
	
	double parse_function(double x,double y){
		return parse_expression2D(Expression,x,y,glob_time);
	}
	
	int i,j,k;
	double x,y,r;
	int d = glob_mesh.size;
	//double* R = random_initial(0.0,0.001,d);
	//initial_conditions_from_image(0,0,"/Home/damage/radszuwe/Desktop/radszuwe/Latex-Dokumente/GAMM2015/figures/data.dat");
	
	for (i=0;i<d;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		r = sqrt((x+0)*(x+0)+(y-0.5)*(y-0.5));
																		
		X[i] = 0;															// ux(x,y,0)
		X[i+d] = 0;															// uy(x,y,0)
		X[i+2*d] = 1.;														// h(x,y,0)					
		//X[i+3*d] = -cos(M_PI*x)*cos(M_PI.*y)>0?1.:-1.;  					// c(x,y,0)
		//X[i+3*d] = sin(M_PI*x)/10.;  
		//X[i+3*d] = 0.001*exp(-12.5*((y-0.5)*(y-0.5)+x*x));								 			
		//X[i+3*d] = (r<0.06) ? -1.:1;
		X[i+3*d] = 0;														// noise
		//X[i+3*d] = tanh(50.*(y-x-0.5));									// 45° interface
		//X[i+3*d] = initial_conditions_from_image(x,y,"compute");
		//X[i+3*d] = 0*tanh(100.*(r-0.1));									// disc
		X[i+4*d] = 0;														// mu(x,y,0)
		X[i+5*d] = T_ref;													// T(x,y,0)	
	}
	//free(R);
	
	if (strcmp(Cond_name,"none") && load_mode!=YES){
		char* Func;
		char Buffer[MAX_FUNC_SIZE];
		int attr_ind;
		for (k=0;k<Initial_conditions.size;k++){
			j = Initial_conditions.Indices[k];
			strcpy(Buffer,Initial_conditions.Cond[k]);
			Func = strtok(Buffer,"=\n");
			if (Func!=NULL){				
				if (strcmp(Func,CONST)==0){
					Func = strtok(NULL,"=");
					for (i=0;i<d;i++) X[i+j*d] = atof(Func);
				}
				else if (strcmp(Func,ATTRIBUTE)==0){							// multi-attributed case not treated !
					Func = strtok(NULL,"=");									// give as many indexes in condition file
					attr_ind = atoi(Func);										// as given in poly file
					
					char Fullname[512];
					int number = 0;
					int attr = 0;
					int** Polys = NULL;
					int* P_sizes = NULL;
					point2D* Seeds = NULL;
					double* Values = NULL;
					sprintf(Fullname,"%s/%s.poly",Mesh_dir,Mesh_name);
					polygons_from_polyfile(Fullname,glob_mesh.Points,&Polys,&P_sizes,&Seeds,&Values,&number,&attr,CHATTY);					
					double* Multi = spread_values2D(&glob_mesh,Polys,P_sizes,Seeds,Values,number,attr);
					copy_vector_content(Multi,X,0,attr_ind*d,d);				// only first index in con file processed
					
					for (i=0;i<number;i++) free(Polys[i]);
					free(Polys);
					free(P_sizes);
					free(Seeds);
					free(Values);
					free(Multi);					
				}
				else if (strcmp(Func,NOISE)==0){					
					Func = strtok(NULL,"=(,)");
					double mean = atof(Func);
					Func = strtok(NULL,"=(,)");
					double sqr = atof(Func);
					double* F = random_initial(mean,sqr,d);				
					copy_vector_content(F,X,0,j*d,d);
					free(F);
				}
				else if (strcmp(Func,IMAGE)==0){
					Func = strtok(NULL,"=\n");
					initial_conditions_from_image(0,0,Func);
					double* F = cartesian_function_on_mesh2D(&initial_conditions_from_image_compute);
					copy_vector_content(F,X,0,j*d,d);
					free(F);					
					initial_conditions_from_image(0,0,"clean");
				}
				else if (strcmp(Func,FUNCTION)==0){					
					Func = strtok(NULL,"=");
					sprintf(Expression,"%s\0",Func);
					double* F = cartesian_function_on_mesh2D(&parse_function);
					copy_vector_content(F,X,0,j*d,d);
					free(F);
				}
				else if (strcmp(Func,SPECIAL)==0){			
					point2D p = {.x=0.326,.y=0.677917};
					double* F = generate_vector(d,1.);
					for (i=0;i<d;i++) if (fabs(p.x-glob_mesh.Points[i].x)<0.0005 && fabs(p.y-glob_mesh.Points[i].y)<0.01) F[i] = 0;
					//(dist(&p,&(glob_mesh.Points[i]))<0.0005) F[i] = 0.;
					p.x = -0.288250;
					p.y = 0.409333;
					for (i=0;i<d;i++) if (fabs(p.y-glob_mesh.Points[i].y)<0.0005 && fabs(p.x-glob_mesh.Points[i].x)<0.01) F[i] = 0;					
					
					copy_vector_content(F,X,0,j*d,d);
					free(F);
				}
			}
			
			if (j==2){				
				force_constraints(&(X[j*d]),0.0,1.0);					// force constraints for damage variable
			}
		}
	}
}

double* get_mobility(double* H,double* C,double* T){
	int i;
	double exp1,exp2,Temp,Temp0;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);

#ifdef MOBILITY_ARRHENIUS
	for (i=0;i<n;i++){
		Temp = R_universal*(T[i]+T_absolute_ref);
		Temp0 = R_universal*T_ref_mobility;
		exp1 = mobility1*exp(-mobility_exp1*(1./Temp-1./Temp0));
		exp2 = mobility2*exp(-mobility_exp2*(1./Temp-1./Temp0));
		Res[i] = ((exp1+exp2)+(exp1-exp2)*linear_limited(C[i]))/2.;
	}
#else
	for (i=0;i<n;i++) Res[i] = ((mobility1+mobility2)+(mobility1-mobility2)*linear_limited(C[i]))/2.;
#endif
	return Res;
}

double* get_mobility_dC(double* H,double* C,double* T){
	int i;
	double exp1,exp2,Temp,Temp0;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
#ifdef MOBILITY_ARRHENIUS
	for (i=0;i<n;i++){
		Temp = R_universal*(T[i]+T_absolute_ref);
		Temp0 = R_universal*T_ref_mobility;
		exp1 = mobility1*exp(-mobility_exp1*(1./Temp-1./Temp0));
		exp2 = mobility2*exp(-mobility_exp2*(1./Temp-1./Temp0));
		Res[i] = (exp1-exp2)*linear_limited_derivative(C[i])/2.;
	}
#else
	for (i=0;i<n;i++) Res[i] = (mobility1-mobility2)*linear_limited_derivative(C[i])/2.;
#endif	
	return Res;
}

double* damage_weight_function(double* H){
	int i;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++) Res[i] = H[i]*(damage_zero_slope+(1.-damage_zero_slope)*H[i]);
	return Res;
}

double* damage_weight_function_dH(double* H){
	int i;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++) Res[i] = damage_zero_slope+2.*(1.-damage_zero_slope)*H[i];
	return Res;
}

double* damage_weight_function_ddH(double* H){
	int n = glob_mesh.size;
	return generate_vector(n,2.*(1.-damage_zero_slope));
}

double* self_strain(double* C,double* T,int order_dc,int order_dT){							
																				// self strain must be a symmetric tensor
	int i,j,k;																	// see also function Self_strain_derivative C and T
	double h,p0,p1,q;
	int n = glob_mesh.size;
	double strain1,strain2,a,b;
	double* Res = zero_vector(n*dimension*dimension);
	
	for (k=0;k<n;k++){
		switch(order_dc){
			case PARTIAL0:
				p0 = 1.;
				p1 = linear_limited(C[k]);
				break;
			case PARTIAL1:
				p0 = 0.;
				p1 = linear_limited_derivative(C[k]);
				break;
			case PARTIAL2:
				p0 = 0;
				p1 = linear_limited_derivative2(C[k]);
				break;
			default:
				p0 = 1.;
				p1 = linear_limited(C[k]);
		}
		switch(order_dT){
			case PARTIAL0: q = T[k]-T_ref;break;
			case PARTIAL1: q = 1.;break;
			case PARTIAL2: q = 0.;break;
			default: q = T[k]-T_ref;
		}		
		
		strain1 = thermal_expansion1.xx;
		strain2 = thermal_expansion2.xx;
		a = (strain1+strain2)/2.;
		b = (strain1-strain2)/2.;
		Res[k] = (a*p0+b*p1)*q;
				
		strain1 = thermal_expansion1.xy;	
		strain2 = thermal_expansion2.xy;
		a = (strain1+strain2)/2.;
		b = (strain1-strain2)/2.;
		Res[n+k] = (a*p0+b*p1)*q;
		Res[2*n+k] = Res[n+k];;
		
		strain1 = thermal_expansion1.yy;
		strain2 = thermal_expansion2.yy;
		a = (strain1+strain2)/2.;
		b = (strain1-strain2)/2.;
		Res[3*n+k] = (a*p0+b*p1)*q;
	}
	return Res;
}

const double llc = 0.9;
const double lta = 1.;

double linear_limited(double c){
	double llb = llc/(1.-llc);
	double lla = exp((llb+1)*log(llc))/llb;
	
	//return tanh(lta*c);																			// tanh-model
	//if (fabs(c)>llc) return (c>llc) ? 1.-lla*exp(-llb*log(c)) : -1.+lla*exp(-llb*log(-c));		// piecewise constinuous
	//else return c;																				// ll(c) = 1-lla/c^llb for c>llc
	return c;
}

double linear_limited_derivative(double c){
	double llb = llc/(1.-llc);
	double lla = exp((llb+1)*log(llc))/llb;
	double csh = cosh(lta*c);																	// tanh-model
	//return lta/(csh*csh);	
	//if (fabs(c)>llc) return lla*llb*exp(-(llb+1.)*log(fabs(c)));								
	//else return 1.;
	return 1.;
}

double linear_limited_derivative2(double c){
	double llb = llc/(1.-llc);
	double lla = exp((llb+1)*log(llc))/llb;
	double tnh = tanh(lta*c);																	// tanh-model
	//return -2.*lta*lta*(1.-tnh*tnh)*tnh;
	//if (fabs(c)>llc) return (c>llc) ? -lla*llb*(llb+1.)*exp(-(llb+2.)*log(c)) : lla*llb*(llb+1.)*exp(-(llb+2.)*log(-c));
	//else return 0;																				// piecewise constinuous
	return 0;
}

modulus2D* zero_modulus(double reg){
	modulus2D* Res = (modulus2D*)malloc(sizeof(modulus2D));
	Res->xxxx = 0;
	Res->yyyy = 0;
	Res->xxyy = 0;
	Res->xxxy = 0;
	Res->yyyx = 0;
	Res->xyxy = 0;
	Res->reg = reg;
	return Res;
}

void insert_modulus_in_vector(modulus2D* C,double* V,int i){
	int n = glob_mesh.size;
	
	V[i] = C->xxxx;
	V[n+i] = C->xxxy;
	V[2*n+i] = C->xxxy;
	V[3*n+i] = C->xxyy;
	
	V[4*n+i] = C->xxxy;
	V[5*n+i] = C->xyxy;
	V[6*n+i] = C->xyxy;
	V[7*n+i] = C->yyyx;
	
	V[8*n+i] = C->xxxy;
	V[9*n+i] = C->xyxy;
	V[10*n+i] = C->xyxy;
	V[11*n+i] = C->yyyx;
	
	V[12*n+i] = C->xxyy;
	V[13*n+i] = C->yyyx;
	V[14*n+i] = C->yyyx;
	V[15*n+i] = C->yyyy;
} 

modulus2D* moduli_sum(modulus2D* C1,modulus2D* C2,double w1,double w2){
	modulus2D* Res = (modulus2D*)malloc(sizeof(modulus2D));
	Res->xxxx = w1*C1->xxxx+w2*C2->xxxx;
	Res->yyyy = w1*C1->yyyy+w2*C2->yyyy;
	Res->xxyy = w1*C1->xxyy+w2*C2->xxyy;
	Res->xxxy = w1*C1->xxxy+w2*C2->xxxy;
	Res->yyyx = w1*C1->yyyx+w2*C2->yyyx;
	Res->xyxy = w1*C1->xyxy+w2*C2->xyxy;
	Res->reg = ((w1*C1->reg)<(w2*C2->reg)) ? w1*C1->reg : w2*C2->reg;
	return Res;
}

modulus2D* moduli_add(modulus2D* C1,modulus2D* C2,double w){
	modulus2D* Res = (modulus2D*)malloc(sizeof(modulus2D));
	C1->xxxx += w*C2->xxxx;
	C1->yyyy += w*C2->yyyy;
	C1->xxyy += w*C2->xxyy;
	C1->xxxy += w*C2->xxxy;
	C1->yyyx += w*C2->yyyx;
	C1->xyxy += w*C2->xyxy;
	C1->reg = ((C1->reg)<(C2->reg)) ? C1->reg : C2->reg;
	return Res;
}

void moduli_mult(modulus2D* C,double w){
	C->xxxx *= w;
	C->yyyy *= w;
	C->xxyy *= w;
	C->xxxy *= w;
	C->yyyx *= w;
	C->xyxy *= w;
}

/*modulus2D* trace_part(modulus2D* C){
	modulus2D* Dev = (modulus2D*)malloc(sizeof(modulus2D));
	Dev->xxxx = (3.*C->xxxx-C->yyyy+2.*C->xxyy)/4.;
	Dev->yyyy = (3.*C->yyyy-C->xxxx+2.*C->xxyy)/4.;
	Dev->xxyy = (C->xxxx+C->yyyy+2.*C->xxyy)/2.;
	Dev->xxxy = (C->xxxy+C->yyyx)/2.;
	Dev->yyyx = (C->xxxy+C->yyyx)/2.;
	Dev->xyxy = 0;
	Dev->reg = C->reg;
	return Dev;
}*/

modulus2D* trace_part(modulus2D* C){
	modulus2D* Dev = (modulus2D*)malloc(sizeof(modulus2D));
	double tr = (C->xxxx+2.*C->xxyy+C->yyyy)/4.;
	
	Dev->xxxx = tr;
	Dev->yyyy = tr;
	Dev->xxyy = tr;
	Dev->xxxy = 0;
	Dev->yyyx = 0;
	Dev->xyxy = 0;
	Dev->reg = C->reg;
	return Dev;
}

modulus2D* deviator(modulus2D* C){
	modulus2D* Tr = trace_part(C);
	modulus2D* Dev = moduli_sum(C,Tr,1.,-1.);
	free(Tr);
	return Dev;
}

void set_moduli_phase(int phase,int part){
	if (Elastic_moduli!=NULL) free(Elastic_moduli);
	switch(phase){
		case 1:
			if (part==DEVIATOR) Elastic_moduli = trace_part(&Moduli_1);
			else Elastic_moduli = deviator(&Moduli_1);
			break;
		case 2: 
			if (part==DEVIATOR) Elastic_moduli = trace_part(&Moduli_2);
			else Elastic_moduli = deviator(&Moduli_2);
			break;
		default: Elastic_moduli = NULL;
	}
}

double damage_anisotropy(double trace){
	return (trace>0) ? 1: 0;
	//return (1.+tanh(10000.*trace))/2.;
	//return 1.;
}

void set_moduli_isotropic(modulus2D* C,double mu,double lambda){
	C->xxxx = 2.*mu+lambda;
	C->yyyy = 2.*mu+lambda;
	C->xxyy = lambda;
	C->xxxy = 0;
	C->yyyx = 0;
	C->xyxy = mu;
	C->reg = (mu_1<mu_2) ? mu_1 : mu_2;
}

void set_moduli_cubic(modulus2D* C,double E1,double E2,double E3){
	C->xxxx = 2.*E1;
	C->yyyy = 2.*E1;
	C->xxyy = E2;
	C->xxxy = 0;
	C->yyyx = 0;
	C->xyxy = 2.*E3;
	C->reg = E1;
}

void insert_moduli(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){	
	modulus2D* E = Elastic_moduli;	
	(B->Len[i]) += 4;																													
	
	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-4] = E->xxxx*v->xx+E->xxxy*(v->xy+v->yx)+E->xyxy*v->yy;				//xx
	B->Indices1[i][len-4] = j;
	B->Indices2[i][len-4] = k;
	B->Values[i][len-3] = E->xxxy*v->xx+E->xxyy*v->xy+E->xyxy*v->yx+E->yyyx*v->yy;			//xy															//xy	
	B->Indices1[i][len-3] = j;
	B->Indices2[i][len-3] = k+d;
	B->Values[i][len-2] = E->xxxy*v->xx+E->xyxy*v->xy+E->xxyy*v->yx+E->yyyx*v->yy;			//yx
	B->Indices1[i][len-2] = j+d;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = E->xyxy*v->xx+E->yyyx*(v->xy+v->yx)+E->yyyy*v->yy;				//yy
	B->Indices1[i][len-1] = j+d;															
	B->Indices2[i][len-1] = k+d;
}

void insert_moduli_bound(sparse_matrix3D* B,matrix2D* v,int i,int j,int k,int d){	
	modulus2D* E = Elastic_moduli;	
	(B->Len[i]) += 2;																													
	(B->Len[i+d]) += 2;																														

	int len = B->Len[i];															
	B->Values[i] = (double*)realloc(B->Values[i],len*sizeof(double));
	B->Indices1[i] = (int*)realloc(B->Indices1[i],len*sizeof(int));
	B->Indices2[i] = (int*)realloc(B->Indices2[i],len*sizeof(int));
	B->Values[i][len-2] = E->xxxx*v->xx+E->xxxy*(v->xy+v->yx)+E->xyxy*v->yy;				//xx
	B->Indices1[i][len-2] = j;
	B->Indices2[i][len-2] = k;
	B->Values[i][len-1] = E->xxxy*v->xx+E->xyxy*v->yx+E->xxyy*v->xy+E->yyyx*v->yy;			//xy															//xy	
	B->Indices1[i][len-1] = j+d;
	B->Indices2[i][len-1] = k;
	
	len = B->Len[i+d];	
	B->Values[i+d] = (double*)realloc(B->Values[i+d],len*sizeof(double));
	B->Indices1[i+d] = (int*)realloc(B->Indices1[i+d],len*sizeof(int));
	B->Indices2[i+d] = (int*)realloc(B->Indices2[i+d],len*sizeof(int));
	B->Values[i+d][len-2] = E->xxxy*v->xx+E->xxyy*v->yx+E->xyxy*v->xy+E->yyyx*v->yy;		//yx
	B->Indices1[i+d][len-2] = j;
	B->Indices2[i+d][len-2] = k;
	B->Values[i+d][len-1] = E->xyxy*v->xx+E->yyyx*(v->xy+v->yx)+E->yyyy*v->yy;				//yy
	B->Indices1[i+d][len-1] = j+d;															
	B->Indices2[i+d][len-1] = k;
}

double* total_strain_trace(double* U,double* C, double* T){
	int k;
	
	int n = glob_mesh.size;
	
	set_var_number2D(1);
	sparse_matrix* Div = set_matrix_Aij_01_2D(0,0,&insert_A_aV_a);
	double* Self = self_strain(C,T,PARTIAL0,PARTIAL0);
	vector_add(&(Self[0]),&(Self[3*n]),1.,n);
	
	double* b = sparse_mult(Div,U);
	linear_map(b,-1.,ID_damage,Self);
	
	double* Res = zero_vector(n);
	multifrontal_solver(ID_damage,Res,b,"total_strain_trace",&UMFPACK_time_interpolation);

	free_sparse(Div);
	free(Self);
	free(b);
	
	int m = 0;
	for (k=0;k<m;k++){
		double* S = neighbour_smoother(Res);
		copy_vector_content(S,Res,0,0,n);
		free(S);
	}
	
	
	/*double* V = set_vector_Ui_0_2D(1);
	double* E0 = linear_strain_tensor(U);	
	double* E = total_strain(E0,C,T,PARTIAL0,PARTIAL0);
	
	double* Res = zero_vector(n);
	for (k=0;k<n;k++) Res[k] = (E[k]+E[3*n+k])/V[k];
	
	free(V);
	free(E);
	free(E0);*/
	
	
	return Res;	
}

double* filtered_total_strain_trace(double* U,double* H,double* C, double* T){
	
	const double factor = .5;
	
	int k;
	double kappa,K;
	
	int n = glob_mesh.size;
	
	//double* V = set_vector_Ui_0_2D(1);
	double* E0 = linear_strain_tensor(U);	
	double* E = total_strain(E0,C,T,PARTIAL0,PARTIAL0);
	double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
	
	double* b = zero_vector(n);
	for (k=0;k<n;k++) b[k] = (E[k]+E[3*n+k]);
	
	double* D = zero_vector(n);
	for (k=0;k<n;k++){
		kappa = (Kappa[k]+Kappa[3*n+k])/2.;
		K = sqrt(kappa/(2*xi))*factor;
		D[k] = K*K*(1.-H[k]);
	}
	
	sparse_matrix* Filter = sparse3D_vB(T_Flux,D,n);
	sparse_add(Filter,ID_damage,1.);
	
	double* Res = zero_vector(n);
	multifrontal_solver(Filter,Res,b,"total_strain_trace",&UMFPACK_time_interpolation);
	
	free_sparse(Filter);
	free(Kappa);
	free(E);
	free(E0);
	free(D);
	free(b);
	return Res;
}

double* strain_det(double* U){
	int k;
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(4);
	double* E = linear_strain_tensor(U);
	vector_pseudo_div(E,V,4*n);	
	double* Res = zero_vector(n);
	for (k=0;k<n;k++) Res[k] = (1.+E[k])*(1.+E[3*n+k])-E[n+k]*E[2*n+k]-1.;
	
	free(V);
	free(E);	
	return Res;	
}


double* total_strain_det(double* U,double* C, double* T){
	int k;
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	
	double* E0 = linear_strain_tensor(U);	
	double* E = total_strain(E0,C,T,PARTIAL0,PARTIAL0);
	
	double* Res = zero_vector(n);
	
	for (k=0;k<n;k++) Res[k] = (E[k]*E[3*n+k]-E[n+k]*E[2*n+k])/(V[k]*V[k]);
	
	free(V);
	free(E);
	free(E0);
	return Res;	
}

void total_strain_eigensystem(double* U,double* C, double* T,double* N1,double* N2,double* Eig){			// Eig holds eigenvalues, size 2*n
	const double tol = 1e-8;													 						// N holds eigenvector of Eig[0]
																										// N1,N2 orthormal
	int k;
	double tr,det,rad,d;
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(dimension*dimension);
	double* E0 = linear_strain_tensor(U);	
	double* E = total_strain(E0,C,T,PARTIAL0,PARTIAL0);
	
	double* Exx = zero_vector(n);
	multifrontal_solver(ID_damage,Exx,&(E[0]),"total_strain_eigensystem",&UMFPACK_time_interpolation);
	double* Exy = zero_vector(n);
	multifrontal_solver(ID_damage,Exy,&(E[n]),"total_strain_eigensystem",&UMFPACK_time_interpolation);
	double* Eyy = zero_vector(n);
	multifrontal_solver(ID_damage,Eyy,&(E[3*n]),"total_strain_eigensystem",&UMFPACK_time_interpolation);
	copy_vector_content(Exx,E,0,0,n);
	copy_vector_content(Exy,E,0,n,n);
	copy_vector_content(Exy,E,0,2*n,n);
	copy_vector_content(Eyy,E,0,3*n,n);
	free(Exx);
	free(Exy);
	free(Eyy);
	//vector_pseudo_div(E,V,dimension*dimension*n);
	
	for (k=0;k<n;k++){
		if (fabs(E[n+k]+E[2*n+k])<tol){
			Eig[k] = E[k];
			Eig[n+k] = E[3*n+k];
			N1[k] = 1.;
			N1[n+k] = 0;
			N2[k] = 0;
			N2[n+k] = 1.;
		}
		else{
			tr = E[k]+E[3*n+k];
			det = E[k]*E[3*n+k]-E[n+k]*E[2*n+k];
			if (tr*tr<4.*det) {
				printf("\nsomething is wrong with strain eigenvalues at index: %d (tr=%e, det=%e\n) -> abort",k,tr,det);
				exit(0);
			}
			rad = sqrt(tr*tr-4.*det);
			Eig[k] = (tr+rad)/2.;
			Eig[n+k] = (tr-rad)/2.;
			
			N1[k] = 1.;
			N1[n+k] = (Eig[k]-E[k])/E[n+k];
			d = sqrt(1.+N1[n+k]*N1[n+k]);
			N1[k] /= d;
			N1[n+k] /= d;
			N2[k] = -N1[n+k];
			N2[n+k] = N1[k];
		}
	}
	
	free(E0);
	free(E);
	free(V);
}

void set_total_strain_eigensystem(double* U,double* C, double* T,eigen_field** Field){
	int n = glob_mesh.size;
	double* N1 = zero_vector(2*n);
	double* N2 = zero_vector(2*n);
	double* Eig = zero_vector(2*n);
	total_strain_eigensystem(U,C,T,N1,N2,Eig);
	init_2D_eigen_field(Field,N1,N2,Eig);
	free(N1);
	free(N2);
	free(Eig);
}

void test_modulus(modulus2D* E){
	if (E->xxyy<0 || E->xyxy<0) printf("Warning negative elastic modulus enocuntered!\n");
}

int indmap(int a,int b,int c,int d){
	int n = glob_mesh.size;
	return (8*a+4*b+2*c+d)*n;
}

double* elastic_constitutive_trace(double* U,double* H,double* C,double* T,int order_dz,int order_dc){	// based on symmetric traceless
	int i,j,k;
	double h1,one_c,one_z,de,tr;
	double* G;
	double* Trace;
	double* Det;
	modulus2D* E;
	modulus2D* Tr;
	modulus2D* Dev;
	
	int n = glob_mesh.size;
	if (Glob_Pointer_Trace0==NULL) Trace = filtered_total_strain_trace(U,H,C,T); else Trace = Glob_Pointer_Trace0;
	//Det = total_strain_det(U,C,T);
	
	// set demage function
	switch (order_dz){
		case PARTIAL0: 
			G = damage_weight_function(H);
			one_z = 1.;
			break;
		case PARTIAL1:
			G = damage_weight_function_dH(H);
			one_z = 0;
			break;
		case PARTIAL2: 
			G = damage_weight_function_ddH(H);
			one_z = 0;
			break;
		default:	// applies for FACTOR (-1)
			G = generate_vector(n,1.);
			one_z = 0;
	}
	
	// set moduli
	double* Res = zero_vector(16*n);
	for (i=0;i<n;i++){
		switch(order_dc){
			case PARTIAL0: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
				break;
			case PARTIAL1: 
				h1 = linear_limited_derivative(C[i]);
				one_c = 0;
				break;
			case PARTIAL2: 
				h1 = linear_limited_derivative2(C[i]);
				one_c = 0;
				break;
			default: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
		}
		E = moduli_sum(&Moduli_1,&Moduli_2,(one_c+h1)/2.,(one_c-h1)/2.);			
		Tr = trace_part(E);
		Dev = deviator(E);
		free(E);
		
		tr = damage_anisotropy(Trace[i]);
		
		//de = damage_anisotropy(Det[i]);
		//E = moduli_sum(Tr,Dev,G[i]*tr+one_z*(1.-tr),G[i]*(1.-de)+one_z*de);		
		//E = moduli_sum(Tr,Dev,G[i]*tr+one_z*(1.-tr),G[i]*tr+(G[i]*(1.-s)+s*one_z)*(1.-tr));
		
		E = moduli_sum(Tr,Dev,G[i]*tr+one_z*(1.-tr),G[i]*tr+one_z*(1.-tr));
		//E = moduli_sum(Tr,Dev,G[i]*tr+one_z*(1.-tr),G[i]);
		insert_modulus_in_vector(E,Res,i);
		
		free(E);
		free(Tr);
		free(Dev);
	}
	
	if (Glob_Pointer_Trace0==NULL) free(Trace);
	//free(Det);
	free(G);
	return Res;
}

double* elastic_constitutive_default(double* U,double* H,double* C,double* T,int order_dz,int order_dc){
	int i,j,k;
	double h1,one_c,one_z,d;
	double* G;
	modulus2D* E;
	
	int n = glob_mesh.size;

	// set demage function
	switch (order_dz){
		case PARTIAL0: 
			G = damage_weight_function(H);
			one_z = 1.;
			break;
		case PARTIAL1:
			G = damage_weight_function_dH(H);
			one_z = 0;
			break;
		case PARTIAL2: 
			G = damage_weight_function_ddH(H);
			one_z = 0;
			break;
		default:	// applies for FACTOR (-1)
			G = generate_vector(n,1.);
			one_z = 0;
	}
	
	// set moduli
	double* Res = zero_vector(16*n);
	for (i=0;i<n;i++){
		switch(order_dc){
			case PARTIAL0: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
				break;
			case PARTIAL1: 
				h1 = linear_limited_derivative(C[i]);
				one_c = 0;
				break;
			case PARTIAL2: 
				h1 = linear_limited_derivative2(C[i]);
				one_c = 0;
				break;
			default: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
		}
		
		E = moduli_sum(&Moduli_1,&Moduli_2,(one_c+h1)/2.,(one_c-h1)/2.);	
		moduli_mult(E,G[i]);
		insert_modulus_in_vector(E,Res,i);	
		free(E);
	}
	
	free(G);
	return Res;
}

/*double* elastic_constitutive_dz_trace(double* U,double* H,double* C,double* T){
	int i,j,k;
	double h1,one_c,one_z,d;
	double* G;
	double* Trace;
	modulus2D* E;
	modulus2D* Tr;
	
	int n = glob_mesh.size;
	if (Glob_Pointer_Trace0==NULL) Trace = total_strain_trace(U,C,T); else Trace = Glob_Pointer_Trace0;
	G = damage_weight_function_dH(H);
	
	// set moduli
	double* Res = zero_vector(16*n);
	for (i=0;i<n;i++){
		h1 = linear_limited(C[i]);
		one_c = 1.;
		one_z = 0;
		E = moduli_sum(&Moduli_1,&Moduli_2,(one_c+h1)/2.,(one_c-h1)/2.);	
		Tr = trace_part(E);
		d = damage_anisotropy(Trace[i]);
		moduli_mult(Tr,G[i]*d+one_z*(1.-d));
		insert_modulus_in_vector(Tr,Res,i);
		
		free(E);
		free(Tr);
	}
	
	if (Glob_Pointer_Trace0==NULL) free(Trace);
	free(G);
	return Res;
}

double* elastic_constitutive_dz_dev(double* U,double* H,double* C,double* T){
	int i,j,k;
	double h1,one_c,d;
	double* G;
	double* Trace;
	modulus2D* E;
	modulus2D* Dev;
	
	int n = glob_mesh.size;
	if (Glob_Pointer_Trace0==NULL) Trace = total_strain_trace(U,C,T); else Trace = Glob_Pointer_Trace0;
	G = damage_weight_function_dH(H);
	
	// set moduli
	double* Res = zero_vector(16*n);
	for (i=0;i<n;i++){
		h1 = linear_limited(C[i]);
		one_c = 1.;
		E = moduli_sum(&Moduli_1,&Moduli_2,(one_c+h1)/2.,(one_c-h1)/2.);	
		Dev = deviator(E);
		d = damage_anisotropy(Trace[i]);
		moduli_mult(Dev,G[i]*d);
		insert_modulus_in_vector(Dev,Res,i);
		
		free(E);
		free(Dev);
	}
	
	if (Glob_Pointer_Trace0==NULL) free(Trace);
	free(G);
	return Res;
}*/


double* elastic_constitutive_eigen(double* U,double* H,double* C,double* T,int order_dz,int order_dc){	// based on eigenprojections	
	int i,j,k;
	double nx,ny,h1,h2,one_c,one_z;
	double* P1;
	double* P2;
	double* G;
	modulus2D* E;
	
	int n = glob_mesh.size;
	
	// set demage function
	switch (order_dz){
		case PARTIAL0: 
			G = damage_weight_function(H);
			one_z = 1.;
			break;
		case PARTIAL1:
			G = damage_weight_function_dH(H);
			one_z = 0;
			break;
		case PARTIAL2: 
			G = damage_weight_function_ddH(H);
			one_z = 0;
			break;
		default:	// applies for FACTOR (-1)
			G = generate_vector(n,1.);
			one_z = 0;
	}
	
	// set moduli
	double* Moduli = zero_vector(16*n);
	for (i=0;i<n;i++){
		switch(order_dc){
			case PARTIAL0: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
				break;
			case PARTIAL1: 
				h1 = linear_limited_derivative(C[i]);
				one_c = 0;
				break;
			case PARTIAL2: 
				h1 = linear_limited_derivative2(C[i]);
				one_c = 0;
				break;
			default: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
		}
		E = moduli_sum(&Moduli_1,&Moduli_2,(one_c+h1)/2.,(one_c-h1)/2.);		
		//test_modulus(E);														// for debugging		
		insert_modulus_in_vector(E,Moduli,i);	
		free(E);
	}
	
	// set strain eigensystem
	double* Eig = zero_vector(2*n);
	double* Vec1 = zero_vector(2*n);
	double* Vec2 = zero_vector(2*n);
	if (Glob_eigensystem0==NULL) total_strain_eigensystem(U,C,T,Vec1,Vec2,Eig);
	else{
		copy_vector_to(Glob_eigensystem0->Eig[0],&(Eig[0]),n);
		copy_vector_to(Glob_eigensystem0->Eig[1],&(Eig[n]),n);
		copy_vector_to(Glob_eigensystem0->Vec[0],Vec1,2*n);
		copy_vector_to(Glob_eigensystem0->Vec[1],Vec2,2*n);
	}
	
	// set projector
	
	double* Projector1 = rank4_tensor_product(Vec1,Vec1,Vec1,Vec1,dimension,n);
	add_rank4_tensor_product(Projector1,Vec1,Vec2,Vec1,Vec2,dimension,n);
	double* Projector2 = rank4_tensor_product(Vec2,Vec2,Vec2,Vec2,dimension,n);
	add_rank4_tensor_product(Projector2,Vec2,Vec1,Vec2,Vec1,dimension,n);
	double* D1 = zero_vector(n);
	double* D2 = zero_vector(n);	
	for (i=0;i<n;i++){
		D1[i] = damage_anisotropy(Eig[i]);
		D2[i] = damage_anisotropy(Eig[n+i]);
	}
	
	free(Vec1);
	free(Vec2);
	free(Eig);
	
	// projection
	double* CP1 = rank4_product(Moduli,Projector1,dimension,n);
	double* CP2 = rank4_product(Moduli,Projector2,dimension,n);
	double* P1CP1 = rank4_product(Projector1,CP1,dimension,n);
	double* P2CP2 = rank4_product(Projector2,CP2,dimension,n);	
	double* P1CP2 = rank4_product(Projector1,CP2,dimension,n);
	double* P2CP1 = rank4_product(Projector2,CP1,dimension,n);
	free(CP1);
	free(CP2);
	
	// get result
	double f1,f2,g;
	double* Res = zero_vector(16*n);
	for (i=0;i<n;i++){
		f1 = D1[i]*G[i]+(1.-D1[i])*one_z;
		f2 = D2[i]*G[i]+(1.-D2[i])*one_z;			
		g = (f1+f2)/2.;
		for (j=0;j<16;j++){
			k = n*j+i;							
			Res[k] = f1*P1CP1[k]+f2*P2CP2[k]+g*(P1CP2[k]+P2CP1[k]);
		}
	}

	//clean
	free(Projector1);
	free(Projector2);
	free(Moduli);
	free(P1CP1);
	free(P2CP2);
	free(P1CP2);
	free(P2CP1);
	free(D1);
	free(D2);
	free(G);
	
	return Res;
}

double* elastic_constitutive(double* U,double* H,double* C,double* T,int order_dz,int order_dc){
	switch(strain_decomposition){
		case EIGENSYSTEM: return elastic_constitutive_eigen(U,H,C,T,order_dz,order_dc);
		case TRACE: return elastic_constitutive_trace(U,H,C,T,order_dz,order_dc);
		default: return elastic_constitutive_default(U,H,C,T,order_dz,order_dc);
	}
	return NULL;
}

/*double* elastic_constitutive(double* U,double* H,double* C,double* T,int order_dz,int order_dc){	
	const double tol = 1e-8;
	int i;
	double nx,ny,nx4,ny4,nx2ny2,h1,h2,one;
	modulus2D* E;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(16*n);
	double* G = NULL;
	
	double* Eig = zero_vector(2*n);
	double* Vec = zero_vector(2*n);
	double* Dummy = zero_vector(2*n);
	total_strain_eigensystem(U,C,T,Vec,Dummy,Eig);
	free(Dummy);
	
	switch (order_dz){
		case PARTIAL0: 
			G = damage_weight_function(H);
			one = 1.;
			break;
		case PARTIAL1:
			G = damage_weight_function_dH(H);
			one = 0;
			break;
		case PARTIAL2: 
			G = damage_weight_function_ddH(H);
			one = 0;
			break;
		default:	// applies for FACTOR (-1)
			G = generate_vector(n,1.);
			one = 0;
	}
	
	double* Projector = zero_vector(16*n);
	for (i=0;i<n;i++){
		nx = Vec[i];
		ny = Vec[n+i];	
		nx4 = nx*nx*nx*nx;
		ny4 = ny*ny*ny*ny;
		nx2ny2 = nx*nx*ny*ny;
		h1 = damage_anisotropy(Eig[i]);
		h2 = damage_anisotropy(Eig[n+i]);		
		 			
		Projector[i] = one-(one-G[i])*(h1*nx4+h2*ny4+2.*sqrt(h1*h2)*nx2ny2);				
		Projector[n+i] = -(one-G[i])*(h1*nx*nx-h2*ny*ny+sqrt(h1*h2)*(-nx*nx+ny*ny))*nx*ny;
		Projector[2*n+i] = Projector[n+i];//-(one-G[i])*(h1*nx*nx-h2*ny*ny+sqrt(h1*h2)*(ny*ny-nx*nx))*nx*ny;
		Projector[3*n+i] = -(one-G[i])*(h1+h2-2.*sqrt(h1*h2))*nx2ny2;
			
		Projector[4*n+i] = Projector[n+i];
		Projector[5*n+i] = one-(one-G[i])*((h1+h2)*nx2ny2+sqrt(h1*h2)*(nx4+ny4));
		Projector[6*n+i] = Projector[3*n+i];
		Projector[7*n+i] = -(one-G[i])*(h1*ny*ny-h2*nx*nx+sqrt(h1*h2)*(nx*nx-ny*ny))*nx*ny;
			
		Projector[8*n+i] = Projector[2*n+i];
		Projector[9*n+i] = Projector[3*n+i];
		Projector[10*n+i] = Projector[5*n+i];//one-(one-G[i])*((h1+h2)*ny*nx*ny*nx+sqrt(h1*h2)*(ny*ny*ny*ny+nx*nx*nx*nx));
		Projector[11*n+i] = Projector[7*n+i];//-(one-G[i])*((h1*ny*ny-h2*nx*nx)+sqrt(h1*h2)*(-ny*ny+nx*nx))*nx*ny;
		
		Projector[12*n+i] = Projector[3*n+i];
		Projector[13*n+i] = Projector[7*n+i];						
		Projector[14*n+i] = Projector[11*n+i];
		Projector[15*n+i] = one-(one-G[i])*(h1*ny4+h2*nx4+2.*sqrt(h1*h2)*nx2ny2);
	}
	free(Eig);
	free(Vec);
	free(G);

	for (i=0;i<n;i++){
		switch(order_dc){
			case PARTIAL0: 
				h1 = linear_limited(C[i]);
				one = 1.;
				break;
			case PARTIAL1: 
				h1 = linear_limited_derivative(C[i]);
				one = 0;
				break;
			case PARTIAL2: 
				h1 = linear_limited_derivative2(C[i]);
				one = 0;
				break;
			default: 
				h1 = linear_limited(C[i]);
				one = 1.;
		}
		
		E = moduli_sum(&Moduli_1,&Moduli_2,(one+h1)/2.,(one-h1)/2.);		
		//Res[n+i] = E->xxxx*xxxy+E->xxxy*(xyxy+yxxy)+E->xxyy*yyxy;
		
		Res[i] = E->xxxx*Projector[i]+E->xxxy*(Projector[4*n+i]+Projector[8*n+i])+E->xxyy*Projector[12*n+i];
		Res[n+i] = E->xxxx*Projector[n+i]+E->xxxy*(Projector[5*n+i]+Projector[9*n+i])+E->xxyy*Projector[13*n+i];
		Res[2*n+i] = E->xxxx*Projector[2*n+i]+E->xxxy*(Projector[6*n+i]+Projector[10*n+i])+E->xxyy*Projector[14*n+i];
		Res[3*n+i] = E->xxxx*Projector[3*n+i]+E->xxxy*(Projector[7*n+i]+Projector[11*n+i])+E->xxyy*Projector[15*n+i];
		
		Res[4*n+i] = E->xxxy*Projector[i]+E->xyxy*(Projector[4*n+i]+Projector[8*n+i])+E->yyyx*Projector[12*n+i];
		Res[5*n+i] = E->xxxy*Projector[n+i]+E->xyxy*(Projector[5*n+i]+Projector[9*n+i])+E->yyyx*Projector[13*n+i];
		Res[6*n+i] = E->xxxy*Projector[2*n+i]+E->xyxy*(Projector[6*n+i]+Projector[10*n+i])+E->yyyx*Projector[14*n+i];
		Res[7*n+i] = E->xxxy*Projector[3*n+i]+E->xyxy*(Projector[7*n+i]+Projector[11*n+i])+E->yyyx*Projector[15*n+i];
		
		Res[8*n+i] = Res[4*n+i];
		Res[9*n+i] = Res[5*n+i];
		Res[10*n+i] = Res[6*n+i];
		Res[11*n+i] = Res[7*n+i];
		
		Res[12*n+i] = E->xxyy*Projector[i]+E->yyyx*(Projector[4*n+i]+Projector[8*n+i])+E->yyyy*Projector[12*n+i];
		Res[13*n+i] = E->xxyy*Projector[n+i]+E->yyyx*(Projector[5*n+i]+Projector[9*n+i])+E->yyyy*Projector[13*n+i];
		Res[14*n+i] = E->xxyy*Projector[2*n+i]+E->yyyx*(Projector[6*n+i]+Projector[10*n+i])+E->yyyy*Projector[14*n+i];
		Res[15*n+i] = E->xxyy*Projector[3*n+i]+E->yyyx*(Projector[7*n+i]+Projector[11*n+i])+E->yyyy*Projector[15*n+i];
		 
		free(E);
	}
	
	free(Projector);
	return Res;
}*/

double* damage_interface_tensor(double* H,double* C,int order_dc){
	int i;
	double one_c,h1;
	matrix2D gamma;
	
	int n = glob_mesh.size;
	double* Res = zero_vector(4*n);
	
	for (i=0;i<n;i++){
		switch(order_dc){
			case PARTIAL0: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
				break;
			case PARTIAL1: 
				h1 = linear_limited_derivative(C[i]);
				one_c = 0;
				break;
			case PARTIAL2: 
				h1 = linear_limited_derivative2(C[i]);
				one_c = 0;
				break;
			default: 
				h1 = linear_limited(C[i]);
				one_c = 1.;
		}
		
		memcpy(&gamma,&interface_tensor1,sizeof(matrix2D));
		mat_mult(&gamma,(one_c+h1)/2.);
		mat_add(&gamma,&interface_tensor2,(one_c-h1)/2.);
		Res[i] = gamma.xx;
		Res[n+i] = gamma.xy;
		Res[2*n+i] = gamma.yx;
		Res[3*n+i] = gamma.yy;		
	}
	
	return Res;
}

void test_rank4_symmetries(double* C){
	
	const double tol=1e-11;
	
	static int calls = 0;
	
	#define X 0
	#define Y 1
	
	int i;
	double v;
	
	calls++;
	int imax = -1;
	double vmax = tol;
	int n = glob_mesh.size;
	double d = (2.*mu_1+lambda_1);
	for (i=0;i<n;i++){
		v = fabs(C[i+indmap(X,Y,X,X)]-C[i+indmap(Y,X,X,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,X,X);
		}	
		v = fabs(C[i+indmap(X,Y,X,Y)]-C[i+indmap(Y,X,X,Y)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,X,Y);
		}
		v = fabs(C[i+indmap(X,Y,Y,X)]-C[i+indmap(Y,X,Y,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,Y,X);
		}
		v = (C[i+indmap(X,Y,Y,Y)]-C[i+indmap(Y,X,Y,Y)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,Y,Y);
		}
		v = (C[i+indmap(X,X,X,Y)]-C[i+indmap(X,X,Y,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,X,X,Y);
		}
		v = (C[i+indmap(X,Y,X,Y)]-C[i+indmap(X,Y,Y,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,X,Y);
		}
		v = (C[i+indmap(Y,X,X,Y)]-C[i+indmap(Y,X,Y,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(Y,X,X,Y);
		}
		v = (C[i+indmap(Y,Y,X,Y)]-C[i+indmap(Y,Y,Y,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(Y,Y,X,Y);
		}
		v = (C[i+indmap(X,X,Y,Y)]-C[i+indmap(Y,Y,X,X)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,X,Y,Y);
		}
		v = (C[i+indmap(X,Y,X,X)]-C[i+indmap(X,X,X,Y)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,X,X);
		}
		v = (C[i+indmap(X,Y,Y,Y)]-C[i+indmap(Y,Y,X,Y)])/d;
		if (v>vmax){
			vmax = v;
			imax = indmap(X,Y,Y,Y);
		}
	}
	
	//if (vmax>tol) 
	printf("test symmetry: error %e at index (%d,%d)\n",vmax,i,imax/n);
	
	#undef X
	#undef Y
}

double* elastic_potential(double* U,double* H,double* C,double* T,int order_dz,int order_dc){	
	int n = glob_mesh.size;
	double* Rank4_modulus = elastic_constitutive(U,H,C,T,order_dz,order_dc);
	double* Res = sparse3D_times_rank4_2D_bilinear(Strain_grad,Rank4_modulus,U,U);
	scalar_mult((double)1/2,Res,n);
	
	free(Rank4_modulus);
	return Res;
}

double* elastic_potential_self(double* U,double* H,double* C,double* T,int order_dz,int order_dc,int order_dT){	
	
	double* A;
	double* B;
	
	int n = glob_mesh.size;
	double* Res = NULL;
	double* Self0 = self_strain(C,T,PARTIAL0,PARTIAL0);
	double* Rank4_modulus0 = elastic_constitutive(U,H,C,T,order_dz,PARTIAL0);
	if (order_dc==PARTIAL0 && order_dT==PARTIAL0){
		Res = sparse3D_times_rank4_matrix_vector(Strain_abs,Rank4_modulus0,Self0,U);	
		double* Order2 = sparse3D_times_rank4_matrix_matrix(T_Source,Self0,Self0,Rank4_modulus0);
		vector_add(Res,Order2,(double)-1/2,n);
		
		free(Order2);
	}
	else if (order_dc==PARTIAL1 && order_dT==PARTIAL0){
		double* dc_Rank4_modulus = elastic_constitutive(U,H,C,T,order_dz,order_dc);
		double* dc_Self = self_strain(C,T,order_dc,order_dT);
		A = sparse3D_times_rank4_matrix_vector(Strain_abs,dc_Rank4_modulus,Self0,U);	
		B = sparse3D_times_rank4_matrix_vector(Strain_abs,Rank4_modulus0,dc_Self,U);	
		vector_add(A,B,1.,n);
		Res = A;
		
		free(B);
		free(dc_Self);
		free(dc_Rank4_modulus);
	}
	else if (order_dc==PARTIAL0 && order_dT==PARTIAL1){
		double* dT_Self = self_strain(C,T,order_dc,order_dT);
		Res = sparse3D_times_rank4_matrix_vector(Strain_abs,Rank4_modulus0,dT_Self,U);	
		
		free(dT_Self);
	}
	else if (order_dc==PARTIAL1 && order_dT==PARTIAL1){		
		double* dc_Rank4_modulus = elastic_constitutive(U,H,C,T,order_dz,PARTIAL1);
		double* dT_Self = self_strain(C,T,PARTIAL0,PARTIAL1);
		double* dcdT_Self = self_strain(C,T,PARTIAL1,PARTIAL1);
		A = sparse3D_times_rank4_matrix_vector(Strain_abs,dc_Rank4_modulus,dT_Self,U);	
		B = sparse3D_times_rank4_matrix_vector(Strain_abs,Rank4_modulus0,dcdT_Self,U);	
		vector_add(A,B,1.,n);
		Res = A;
		
		free(B);
		free(dT_Self);
		free(dcdT_Self);
		free(dc_Rank4_modulus);
	}
	if (order_dc==PARTIAL0 && order_dT==PARTIAL2){
		Res = zero_vector(n);
	}
	
	scalar_mult(-1.,Res,n);
	free(Rank4_modulus0);
	free(Self0);
	return Res;
}

sparse_matrix* elastic_potential_Hesse(double* U,double* H,double* C,double* T,int order_dz,int order_dc){
	sparse_matrix* Res;
	double* Rank4_modulus = elastic_constitutive(U,H,C,T,order_dz,order_dc);
	if (El_Hesse_Map==NULL) Res = sparse3D_times_rank4_2D_right(Strain_grad,Rank4_modulus);
	else Res = sparse3D_times_rank4_2D_right_use_map(Strain_grad,Rank4_modulus,El_Hesse_Map);
	//Res = sparse3D_times_rank4_2D_right(Strain_grad,Rank4_modulus);
	
	free(Rank4_modulus);
	return Res;
}

sparse_matrix* elastic_potential_du(double* U,double* H,double* C,double* T,int order_dz,int order_dc){
	sparse_matrix* Res;
	double* Rank4_modulus = elastic_constitutive(U,H,C,T,order_dz,order_dc);
	Res = sparse3D_times_rank4_2D_middle(Strain_grad,Rank4_modulus,U);
	
	free(Rank4_modulus);
	return Res;
}

double* linear_rank2(double* C_rank4,double* X_rank2){										// only valid in 2D
	int i,j,k,l;
	int n = glob_mesh.size;
	double* Res = zero_vector(4*n);
	for (k=0;k<n;k++){
		for (l=0;l<16;l++){
			i = l / 4;
			j = l % 4;
			Res[n*i+k] += C_rank4[n*l+k]*X_rank2[n*j+k];
		}
	}
	return Res;
}

double* bilinear_rank2(double* X_rank2,double* C_rank4,double* Y_rank2){					// only valid in 2D
	int i,j,k,l;
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	for (k=0;k<n;k++){
		for (l=0;l<16;l++){
			i = l / 4;
			j = l % 4;
			Res[k] += X_rank2[n*i+k]*C_rank4[n*l+k]*Y_rank2[n*j+k];
		}
	}
	return Res;
}

double* constitutive_stress(double* Strain,double* U,double* H,double* C,double* T,int order_dz,int order_dc){	
	double* Rank4_modulus = elastic_constitutive(U,H,C,T,order_dz,order_dc);
	double* Stress = linear_rank2(Rank4_modulus,Strain);
	
	free(Rank4_modulus);
	return Stress;
}

double* Self_strain_div(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	
	sparse_matrix3D* Strain_abs_T = sparse3D_transpose13(Strain_abs,dimension*n);
	double* Self = self_strain(C,T,PARTIAL0,PARTIAL0);
	double* Rank4_modulus = elastic_constitutive(U,H,C,T,PARTIAL0,PARTIAL0);
	double* Res = sparse3D_times_rank4_matrix(Strain_abs_T,Rank4_modulus,Self);
	
	free_sparse3D(Strain_abs_T);
	free(Self);
	free(Rank4_modulus);
	return Res;
	
}

double get_histogram_thres(double* Hist,int size,double max_area,double factor){
	int i;
	
	//double thres = (a_1<a_2) ? log10(a_1*factor) : log10(a_2*factor);
	double thres = log10(factor);
	double area = 0;
	double res = NAN;
	for (i=size-1;i>=0;i--){
		if (Hist[size+i]>thres){
			area += Hist[i];
			if (area<max_area) res = Hist[size+i];else break;			
		}
	}
	return res;
}

void refine_by_gradient(double* X,double* Refine,int field_number,double a_min,double a_max){
		
	const double thr_max = 1e-4;
	
	int i;
	double r;
	
	int n = glob_mesh.size;
	double* Weight = generate_vector(field_number,1.);
	//Weight[0] = 3.;
	//Weight[1] = 3.;
	double total_area = get_total_area();	
	
	double* Grad = max_steepnes_2D(X,Weight,field_number);
	for (i=0;i<n;i++){
		r = 2.*front_resolution*front_resolution*Grad[i]*Grad[i];
		if (r!=0){					
			if (1./r<thr_max) Refine[i] = a_min; else Refine[i] = a_max;
			
			/*Refine[i] = (Refine[i]<a_min) ? a_min : Refine[i];
			Refine[i] = (Refine[i]>a_max) ? Refine[i] = a_max: Refine[i];*/
		}
		else Refine[i] = a_max;
	}
	//scalar_mult(total_area,Refine,n);
	free(Weight);
	free(Grad);
}

void refine_by_strain_energy(double* X,double* Refine,double a_min){
	int i;
	
	int n = glob_mesh.size;
	double* U = &(X[0]);
	double* H = &(X[dimension*n]);
	double* C = &(X[(dimension+1)*n]);
	double* T = &(X[(dimension+3)*n]);
	
	double* Wel = compute_strain_energy(U,H,C,T,PARTIAL0);
	double* Alpha = damage_rate_independent(C);
	vector_pseudo_div(Wel,Alpha,n);
	double* Hist = Field_histogramm(Wel,histogram_resolution);
	double exp10 = get_histogram_thres(Hist,histogram_resolution,strain_ref_max_area*get_total_area(),strain_ref_alpha_factor);
	if (isnan(exp10)){
		printf("Warning: get_histogram_thres returned NAN -> skip refinement\n");
		free(Wel);
		free(Hist);
		free(Alpha);
		return;
	}
	
	double thres = pow(10.,exp10);
	//printf("refine threshold: %e\n",thres);
	
	for (i=0;i<n;i++) if (fabs(Wel[i])>thres && fabs(Refine[i])>a_min) Refine[i] = a_min;
	
	/*char Buffer[512];
	sprintf(Buffer,"%s/%s/hist_%d",Output_dir,Output_name,glob_time_index);
	print_histogram(Buffer,Hist,res);*/
	
	free(Wel);
	free(Hist);
	free(Alpha);
}

void refine_by_damage_predictor(double* X,double* Refine,double a_min){
	const double Htol = 1e-6;														// tolerance for damage to be considered as zero
	const double Hthres2 = 0.95;													// damage threshold for second order predictor
	
	int i;
	double* Hpred;
	
	int n = glob_mesh.size;
	double* U = &(X[0]);
	double* H = &(X[dimension*n]);
	double* C = &(X[(dimension+1)*n]);
	double* T = &(X[(dimension+3)*n]);
	
	if (xi==0){
		Hpred = damage_predictor(U,H,C,T); 
		for (i=0;i<n;i++){		
			if (Hpred[i]<0 && H[i]>Htol) Refine[i] = a_min;	
		}
	}
	else{
		Hpred = secondorder_predictor(U,H,C,T);
		for (i=0;i<n;i++){		
			if (Hpred[i]<(Hthres2-1.)) Refine[i] = a_min;	
		}
		
	}
	free(Hpred);
}

double gauss(point2D* P1,point2D* P2){
	static double sigma = 1.;
	if (P1==NULL || P2==NULL){															
		sigma = sqrt(2.*get_largest_element_area(&glob_mesh,&elements))/8.;
		return 0;
	}
	else return exp(-((P2->x-P1->x)*(P2->x-P1->x)+(P2->y-P1->y)*(P2->y-P1->y))/(2.*sigma*sigma));
}

void refine_by_future_value(double* H_future,double* Refine,double a_min){
	const double Hthres = 0.5;
	
	double d;
	int i,j;
	
	if (H_future!=NULL){
		int n = glob_mesh.size;			
		//double l = sqrt(2.*get_largest_element_area(&base_mesh,&base_elements));
		double l = sqrt(2.*min_refine_area);
		double range = 5.*l;
		
		// take minimum in range
		double* Min = zero_vector(n);
		range *= range;
		for (i=0;i<n;i++){
			Min[i] = H_future[i];
			for (j=0;j<n;j++){
				d = sqrdist(&(glob_mesh.Points[i]),&(glob_mesh.Points[j]));			
				if (d<range){
					if (H_future[j]<Min[i]) Min[i] = H_future[j];
				}
			}
		}
		
		for (i=0;i<n;i++) if (Min[i]<Hthres) Refine[i] = a_min;
		free(Min);
		
		// with continous convolution
		/*gauss(NULL,NULL);																		// initialize
		
		double* F = generate_vector(n,1.);
		vector_add(F,H_future,-1.,n);
		sparse_matrix* A = get_convolution_matrix(&gauss,range);					
		double* Convolution = sparse_mult(A,F);
		
		for (i=0;i<n;i++) if (Convolution[i]>Hthres) Refine[i] = a_min*total_area;
		
		free_sparse(A);
		free(F);
		free(Convolution);*/
	}
}

void refine_by_damage_value(double* X,double* Refine,double a_min){
	//const double Hthres = 0.6;														// damage threshold for refinement
	
	int i;
	
	int n = glob_mesh.size;	
	double* H = &(X[dimension*n]);
	
	for (i=0;i<n;i++){	
		if (H[i]<crack_thres) Refine[i] = a_min;
	}
}

void refine_by_damage_change(int time_index,double* Refine,double a_min){
	const double thr_c = 2e-4;
	const double thr_d = 5e+1;
	const double l_fraction = .2;
	
	static int last_index = 1;
	static double* Last = NULL;
	
	int i,n,n_prev;
	double* H;
	double* H_prev;
	double* Last_dH;
	
	
	if (time_index>1){	
		
		n = Glob_mesh_sizes[time_index];
		n_prev = Glob_mesh_sizes[time_index-1];
		H = clone_vector(All_H[time_index],n);
		if (time_index==last_index+1){												// use previous interpolation matrix
			H_prev = interpolated(All_H[time_index-1],1,n_prev,NULL,NULL);	
			if (Last!=NULL) Last_dH = interpolated(Last,1,n_prev,NULL,NULL);	
		}
		else{
			mesh2D old_mesh;														// create new interpolation
			element_collection old_elements;
			set_var_number2D(1);
			Load_mesh(glob_mesh_index-1,&old_mesh,&old_elements,QUIET);		
			sparse_matrix* Left = set_matrix_Aij_00_2D(0,0,&insert_AV);
			sparse_matrix* Right = Get_interpolation_matrix(&glob_mesh,&old_mesh,&elements,&old_elements);			
			H_prev = interpolated(All_H[time_index-1],1,n_prev,Left,Right);
			if (Last!=NULL) Last_dH = interpolated(Last,1,n_prev,Left,Right);
			FreeMesh(&old_mesh);												
			free(old_elements.Elements);
			free_sparse(Left);
			free_sparse(Right);
		}
		double l = sqrt(2.*get_largest_element_area(&base_mesh,&base_elements));
		gauss(NULL,NULL);
		sparse_matrix* A = get_convolution_matrix(&gauss,l_fraction*l);
		double* R = zero_vector(n);
		for (i=0;i<n;i++){
			double log0 = (H_prev[i]>0) ? log10(H_prev[i]) : -16.;
			double log1 = (H[i]>0) ? log10(H[i]) : -16.;
			if (log0<-8.) R[i] = 0;
			else{
				if (log0-log1>1.) R[i] = 1.;				
			}
		}
		if (Last!=NULL){
			vector_add(R,Last_dH,3.,n);
			scalar_mult((double)1/4,R,n);
			free(Last_dH);
			free(Last);
		}		
		Last = clone_vector(R,n);	
		
		//char Name[512];
		//sprintf(Name,"/Home/damage/radszuwe/Daten/record_%d",glob_mesh_index);
		//print_scalar_data(Name,Last,n);
		
		double* dH = sparse_mult(A,R);		
		for (i=0;i<n;i++) if (fabs(dH[i])>thr_c) Refine[i] = a_min;
				
		last_index = time_index;
	
		free_sparse(A);
		free(R);
		free(H);
		free(dH);
		free(H_prev);
	}
}

// general functions /////////////////////////////////////////////////////////////////////

static void signal_handler(int sig){
	SIGINT_flag = 1;
}

void read_mesh_2D(char* Dir,char* Name,mesh2D* Mesh,element_collection* Triangles,double** Attributes,int* attr_num,int quiet){
	int element_number;
	if (!quiet) printf("read mesh from file %s/%s\n\n",Dir,Name);
	if (Mesh==NULL) Mesh = &glob_mesh;
	if (Triangles==NULL) Triangles = &elements;
	element_number = 0;
	read_node_file2D(Dir,Name,Mesh,Attributes,attr_num,quiet);
	read_element_file2D(Dir,Name,Mesh,&(Triangles->Elements),&element_number,quiet);
	Triangles->size = element_number;
	Sort_knots(Mesh);
	create_look_up_table(Mesh);
}

sparse_matrix* Get_interpolation_matrix(mesh2D* New_mesh,mesh2D* Old_mesh,element_collection* New_elements,element_collection* Old_elements){
	sparse_matrix* Res;
	double max_old,max_new;
	max_old = max_triangle_size(Old_mesh,Old_elements);
	if (New_elements==NULL) max_new = 0; else max_new = max_triangle_size(New_mesh,New_elements);
	double d = (max_old > max_new) ? max_old : max_new;
	
	switch(interpolation_method){
		case BARYCENTRIC: Res = get_barycentric_interpolation_matrix(New_mesh,Old_mesh,Old_elements);break;				
		case ELEMENT_BASE: Res = FEM2D_interpolation_matrix(New_mesh,Old_mesh,New_elements,Old_elements,d,interpolation_tol);break;
		default: Res = NULL;
	}
	
	return Res;
}

void get_crack_set(double* H,double tol,int** Nodes,int* size){
	int i,j,k;
	
	int n = glob_mesh.size;
	int N = elements.size;
	int m = base_mesh.size;
	int M = base_elements.size;
	
	*size = 0;
	*Nodes = NULL;
	for (i=0;i<n;i++){
		if (H[i]<tol){
			(*size)++;
			*Nodes = (int*)realloc(*Nodes,(*size)*sizeof(int));
			(*Nodes)[(*size)-1] = i;
		}
		else{
			for (j=0;j<glob_mesh.Sizes[i];j++){
				k = glob_mesh.Connections[i][j];
				if (H[k]<tol){
					(*size)++;
					*Nodes = (int*)realloc(*Nodes,(*size)*sizeof(int));
					(*Nodes)[(*size)-1] = i;
					break;
				}
			}
		}
	}
}

int cmp_progname(char* Name1,char* Name2){
	int i,len1,len2,res;
	char** list1;
	char** list2;
	list1 = split(Name1,"/",&len1);
	list2 = split(Name2,"/",&len2);
	
	res = 0;
	if (len1>0 && len2>0){
		if (strcmp(list1[len1-1],list2[len2-1])==0) res = 1;
	}
	
	for (i=0;i<len1;i++) free(list1[i]);
	for (i=0;i<len2;i++) free(list2[i]);
	free(list1);
	free(list2);
	return res;
}

int Read_param_strings(char* Filename,char*** Names,char*** Values,int* Len){
	const int SIZE = 512;
	int i,fin,status;
	FILE* file;
	char buff[SIZE];
	char** line;
	
	int len = 0;
	int size = 0;
	
	*Names = NULL;
	*Values = NULL;
	*Len = 0;
	file = Open_file(Filename,"r");
	if (file==NULL) exit(0);
	readLine(file,buff,SIZE);
	line = split(buff,"\t",&len);
	if (line==NULL || cmp_progname(line[0],Prog_name)!=1){
		printf("Warning: parameter file does not match program %s!=%s\n",Prog_name,line[0]);
		return FAILED;
	}
	for (i=0;i<len;i++) free(line[i]);
	free(line);
	
	fin = 0;
	size = 0;
	while(readLine(file,buff,SIZE)==1){
		if (buff[0]!='#'){
			line = split(buff,"\t",&len);
			if (len==2){
				size++;
				*Names = (char**)realloc(*Names,size*sizeof(char*));
				*Values = (char**)realloc(*Values,size*sizeof(char*));
				(*Names)[size-1] = (char*)malloc(SIZE*sizeof(char));
				(*Values)[size-1] = (char*)malloc(SIZE*sizeof(char));
				(*Names)[size-1][0] = '\0';
				(*Values)[size-1][0] = '\0';
				strcpy((*Names)[size-1],line[0]);
				strcpy((*Values)[size-1],line[1]);
			}else return FAILED;
			for (i=0;i<len;i++) free(line[i]);
			free(line);
		}
	}
	*Len = size;
	fclose(file);
	return SUCCESS;
}

char* Get_parameter(char* Fullname,char* Param_name){
	const int SIZE = 512;
	int i,fin,len;
	FILE* file;
	char buff[SIZE];
	char** line;
	
	char* Res = NULL;
	file = Open_file(Fullname,"r");
	readLine(file,buff,SIZE);
	line = split(buff,"\t",&len);
	if (cmp_progname(line[0],Prog_name)!=1){
		printf("Warning: parameter file does not match program\n");
		return FAILED;
	}
	for (i=0;i<len;i++) free(line[i]);
	free(line);
	
	fin = 0;
	while(readLine(file,buff,SIZE)==1){
		line = split(buff,"\t",&len);
		if (len==2 && strcmp(line[0],Param_name)==0){
			Res = (char*)malloc(SIZE*sizeof(char));
			sprintf(Res,"%s",line[1]);
			fin = 1;
		}	
		for (i=0;i<len;i++) free(line[i]);
		free(line);
		if (fin==1) break;
	}
	fclose(file);
	return Res;
}

int get_bool(char* expression){
	if (strcmp(expression,"YES")==0) return YES;
	else if (strcmp(expression,"NO")==0) return NO;
	else return -1;
}

char* get_string_bool(int b){
	if (b==YES) return "YES"; else return "NO";
}

int get_solver_number(char* solver_string){
	if (strcmp(solver_string,ACTIVE_SET_STRING)==0) return ACTIVE_SET;
	else if (strcmp(solver_string,REFLECTIVE_NEWTON_STRING)==0) return REFLECTIVE_NEWTON;
	else return 0;
}

char* get_solver_string(int solver_number){
	if (solver_number==ACTIVE_SET) return ACTIVE_SET_STRING;
	else if (solver_number==REFLECTIVE_NEWTON) return REFLECTIVE_NEWTON_STRING;
	else return "no solver chosen";
}

void process_elastic_moduli_arg(char* var,char* input){
	
	if (strcmp(var,P_MU1)==0 || strcmp(var,P_LAMBDA1)==0) Elastic_moduli = &Moduli_1;
	if (strcmp(var,P_MU2)==0 || strcmp(var,P_LAMBDA2)==0) Elastic_moduli = &Moduli_2;
	
	char* left = strpbrk(input,"(");
	char* right = strpbrk(input,")");
	
	if (left==NULL && right==NULL){
	#ifdef ANISOTROPIC_ELASTICITY
		if (strcmp(var,P_LAMBDA1)!=0 && strcmp(var,P_LAMBDA2)!=0){ 
			printf("not enough moduli given (%s)-> abort\n",input);
			exit(0);
		}
	#else
		if (strcmp(var,P_MU1)==0){
			mu_1 = atof(input);
			Elastic_moduli->xxxx += 2.*mu_1;
			Elastic_moduli->yyyy += 2.*mu_1;
			Elastic_moduli->xyxy += mu_1;
		}
		else if (strcmp(var,P_MU2)==0){
			mu_2 = atof(input);
			Elastic_moduli->xxxx += 2.*mu_2;
			Elastic_moduli->yyyy += 2.*mu_2;
			Elastic_moduli->xyxy += mu_2;
		}
		else if (strcmp(var,P_LAMBDA1)==0){
			lambda_1 = atof(input);
			Elastic_moduli->xxxx += lambda_1;
			Elastic_moduli->yyyy += lambda_1;
			Elastic_moduli->xxyy += lambda_1;
		}
		else if (strcmp(var,P_LAMBDA2)==0){
			lambda_2 = atof(input);
			Elastic_moduli->xxxx += lambda_2;
			Elastic_moduli->yyyy += lambda_2;
			Elastic_moduli->xxyy += lambda_2;
		}
	#endif
	}
	else{
		
	#ifndef ANISOTROPIC_ELASTICITY
		printf("wrong parameters: no anisotropy defined in the model -> abort\n");
		exit(0);
	#endif
		
		if (left==NULL || right==NULL) printf("wrong format for %s -> use default value\n",var);
		else{		
			int i,j;	
			int len = 0;
			*right = '\0';			
			char** Parts = split(++left,",",&len);
			
			if (len!=7) printf("Warning: wrong number of elastic parameters, given %d, needed 7\n",len);
			for (i=0;i<len;i++){
				int l = 0;				
				char** Sub = split(Parts[i],"=",&l);
				if (l==2){
					if (strstr(Sub[0],"xxxx")!=NULL) Elastic_moduli->xxxx = atof(Sub[1]);
					else if (strstr(Sub[0],"yyyy")!=NULL) Elastic_moduli->yyyy = atof(Sub[1]);
					else if (strstr(Sub[0],"xxxy")!=NULL) Elastic_moduli->xxxy = atof(Sub[1]);
					else if (strstr(Sub[0],"yyyx")!=NULL) Elastic_moduli->yyyx = atof(Sub[1]);
					else if (strstr(Sub[0],"xxyy")!=NULL) Elastic_moduli->xxyy = atof(Sub[1]);
					else if (strstr(Sub[0],"xyxy")!=NULL) Elastic_moduli->xyxy = atof(Sub[1]);					
					else if (strstr(Sub[0],"reg")!=NULL) Elastic_moduli->reg = atof(Sub[1]);
				}
				for (j=0;j<l;j++) free(Sub[j]);
				free(Sub);
			}
			for (i=0;i<len;i++) free(Parts[i]);
			free(Parts);
			
			/*printf("\nmoduli:\n");
			printf("Cxxxx = %f\n",Moduli_1.xxxx);
			printf("Cyyyy = %f\n",Moduli_1.yyyy);
			printf("Cxxxy = %f\n",Moduli_1.xxxy);
			printf("Cyyyx = %f\n",Moduli_1.yyyx);
			printf("Cxxyy = %f\n",Moduli_1.xxyy);
			printf("Cxyxy = %f\n",Moduli_1.xyxy);
			printf("reg = %f\n",Moduli_1.reg);*/
		}
	}	
}

int process_matrix2D_arg(char* var,char* input,matrix2D* Exp){
	char* left = strpbrk(input,"(");
	char* right = strpbrk(input,")");
	
	if (left==NULL && right==NULL){
		Exp->xx = atof(input);
		Exp->xy = 0;
		Exp->yx = 0;
		Exp->yy = Exp->xx;
		return NO;
	}
	else{	
		if (left==NULL || right==NULL){
			printf("wrong format for %s -> abort\n",var);
			exit(0);
		}
		else{	
			int i,j;	
			int len = 0;
			*right = '\0';			
			char** Parts = split(++left,",",&len);
			
			if (len!=3) printf("Warning: wrong number of expansion parameters, given %d, needed 3\n",len);
			for (i=0;i<len;i++){
				int l = 0;				
				char** Sub = split(Parts[i],"=",&l);
				if (l==2){
					if (strstr(Sub[0],"xx")!=NULL) Exp->xx = atof(Sub[1]);		
					else if (strstr(Sub[0],"xy")!=NULL){Exp->xy = atof(Sub[1]);Exp->yx = Exp->xy;}
					else if (strstr(Sub[0],"yy")!=NULL) Exp->yy = atof(Sub[1]);					
				}
				for (j=0;j<l;j++) free(Sub[j]);
				free(Sub);
			}
			for (i=0;i<len;i++) free(Parts[i]);
			free(Parts);			
		}
		return YES;
	}
}

void process_thermexp_arg(char* var,char* input){
	
	matrix2D* Exp = NULL;
	if (strcmp(var,P_THERMAL_EXPANSION1)==0) Exp = &thermal_expansion1;
	if (strcmp(var,P_THERMAL_EXPANSION2)==0) Exp = &thermal_expansion2;
	if (Exp==NULL) return;
	
	int anisotropic = process_matrix2D_arg(var,input,Exp);
	
#ifndef ANISOTROPIC_ELASTICITY
	if (anisotropic==YES){
		printf("wrong parameters: no anisotropy defined in the model -> abort\n");
		exit(0);
	}	
#endif	
}

void process_kappa_arg(char* var,char* input){
	
	matrix2D* Int = NULL;
	if (strcmp(var,P_KAPPA1)==0) Int = &interface_tensor1;
	if (strcmp(var,P_KAPPA1)==0) Int = &interface_tensor2;
	if (Int==NULL) return;
	
	process_matrix2D_arg(var,input,Int);
}	

void process_robin_arg(char* var,char* input){
	
	matrix2D* Exp;
	if (strcmp(var,P_ROBIN_STIFFNESS)!=0) return;
	
	char* left = strpbrk(input,"(");
	char* right = strpbrk(input,")");
	
	if (left==NULL && right==NULL){
		Robin_stiffness.x = atof(input);
		Robin_stiffness.y = Robin_stiffness.x;
	}
	else{	
		if (left==NULL || right==NULL){
			printf("wrong format for %s -> abort\n",var);
			exit(0);
		}
		else{	
			int i,j;	
			int len = 0;
			*right = '\0';			
			char** Parts = split(++left,",",&len);
			if (len!=2) printf("Warning: wrong number of robin parameters, given %d, needed 2\n",len);
			for (i=0;i<len;i++){
				int l = 0;				
				char** Sub = split(Parts[i],"=",&l);
				if (l==2){
					if (strstr(Sub[0],"x")!=NULL) Robin_stiffness.x = atof(Sub[1]);							
					else if (strstr(Sub[0],"y")!=NULL) Robin_stiffness.y = atof(Sub[1]);					
				}
				for (j=0;j<l;j++) free(Sub[j]);
				free(Sub);
			}
			for (i=0;i<len;i++) free(Parts[i]);
			free(Parts);		
		}
		
	}
}

void print_moduli(FILE* file,char* var){
	modulus2D* E = NULL;
	if (strcmp(P_MU1,var)==0) E = &Moduli_1;
	if (strcmp(P_MU2,var)==0) E = &Moduli_2;
	
	if (E!=NULL){
		fprintf(file,"%s\t(xxxx=%e,yyyy=%e,xxyy=%e,xyxy=%e,xxxy=%e,yyyx=%e,reg=%e)\n"
			,var,E->xxxx,E->yyyy,E->xxyy,E->xyxy,E->xxxy,E->yyyx,E->reg);
	}
}

void print_thermexp(FILE* file,char* var){
	matrix2D* Exp = NULL;
	if (strcmp(P_THERMAL_EXPANSION1,var)==0) Exp = &thermal_expansion1;
	if (strcmp(P_THERMAL_EXPANSION2,var)==0) Exp = &thermal_expansion2;
	
	if (Exp!=NULL){
	#ifdef ANISOTROPIC_ELASTICITY
		fprintf(file,"%s\t(xx=%e,xy=%e,yy=%e)\n",var,Exp->xx,Exp->xy,Exp->yy);
	#else 
		fprintf(file,"%s\t%e\n",var,Exp->xx);
	#endif
	}
}

void print_kappa(FILE* file,char* var){
	matrix2D* Exp = NULL;
	if (strcmp(P_KAPPA1,var)==0) Exp = &interface_tensor1;
	if (strcmp(P_KAPPA2,var)==0) Exp = &interface_tensor2;
	
	if (Exp!=NULL) fprintf(file,"%s\t(xx=%e,xy=%e,yy=%e)\n",var,Exp->xx,Exp->xy,Exp->yy);
}

void print_robin(FILE* file,char* var){
	if (strcmp(P_ROBIN_STIFFNESS,var)==0){
		fprintf(file,"%s\t(x=%e,y=%e)\n",var,Robin_stiffness.x,Robin_stiffness.y);
	}
}

void Assign_param_strings(char** Names,char** Values,int len){
	int i,j;
	if (len>=P_COUNTER){
		for (i=0;i<len;i++){
			if (strcmp(Names[i],P_OUTPUTNAME)==0) sprintf(Output_name,"%s",Values[i]);
			else if (strcmp(Names[i],P_OUTPUTDIR)==0) sprintf(Output_dir,"%s",Values[i]);
			else if (strcmp(Names[i],P_MESHNAME)==0) sprintf(Mesh_name,"%s",Values[i]);
			else if (strcmp(Names[i],P_MESHDIR)==0) sprintf(Mesh_dir,"%s",Values[i]);
			else if (strcmp(Names[i],P_LOAD)==0) sprintf(Load_name,"%s",Values[i]);
			else if (strcmp(Names[i],P_SAMPLE_INTERVAL)==0) sample_interval = atoi(Values[i]);
			else if (strcmp(Names[i],P_COND)==0) sprintf(Cond_name,"%s",Values[i]);
			
			else if (strcmp(Names[i],P_MU1)==0) process_elastic_moduli_arg(P_MU1,Values[i]);//mu_1 = atof(Values[i]);
			else if (strcmp(Names[i],P_MU2)==0) process_elastic_moduli_arg(P_MU2,Values[i]);//mu_2 = atof(Values[i]);
			else if (strcmp(Names[i],P_LAMBDA1)==0) process_elastic_moduli_arg(P_LAMBDA1,Values[i]);//lambda_1 = atof(Values[i]);
			else if (strcmp(Names[i],P_LAMBDA2)==0) process_elastic_moduli_arg(P_LAMBDA2,Values[i]);//lambda_2 = atof(Values[i]);
			else if (strcmp(Names[i],P_ALPHA1)==0) a_1 = atof(Values[i]);
			else if (strcmp(Names[i],P_ALPHA2)==0) a_2 = atof(Values[i]);
			else if (strcmp(Names[i],P_BETA)==0) beta = atof(Values[i]);
			else if (strcmp(Names[i],P_KAPPA1)==0) process_kappa_arg(P_KAPPA1,Values[i]);
			else if (strcmp(Names[i],P_KAPPA2)==0) process_kappa_arg(P_KAPPA2,Values[i]);
			else if (strcmp(Names[i],P_XI)==0) xi = atof(Values[i]);
			else if (strcmp(Names[i],P_ETA_EPS)==0) eta_eps = atof(Values[i]);
			else if (strcmp(Names[i],P_MOBILITY1)==0) mobility1 = atof(Values[i]);
			else if (strcmp(Names[i],P_MOBILITY2)==0) mobility2 = atof(Values[i]);
			else if (strcmp(Names[i],P_KAPPA_C)==0) kappa_c = atof(Values[i]);
			else if (strcmp(Names[i],P_LANDAU_POT)==0) pot_factor = atof(Values[i]);
			else if (strcmp(Names[i],P_VISCOSITY_C)==0) vis_c = atof(Values[i]);			
			else if (strcmp(Names[i],P_T_CRITICAL)==0) T_crit = atof(Values[i]);
			else if (strcmp(Names[i],P_EUTECTIC)==0) c_eutectic = atof(Values[i]);			
			else if (strcmp(Names[i],P_T_INITIAL)==0) T_ref = atof(Values[i]);
			else if (strcmp(Names[i],P_THERMAL_EXPANSION1)==0) process_thermexp_arg(P_THERMAL_EXPANSION1,Values[i]);//thermal_expansion1 = atof(Values[i]);
			else if (strcmp(Names[i],P_THERMAL_EXPANSION2)==0) process_thermexp_arg(P_THERMAL_EXPANSION2,Values[i]);//thermal_expansion2 = atof(Values[i]);
			else if (strcmp(Names[i],P_HEAT_CAPACITY0)==0) rho_0 = atof(Values[i]);
			else if (strcmp(Names[i],P_HEAT_CAPACITY1)==0) rho_1 = atof(Values[i]);
			else if (strcmp(Names[i],P_HEAT_CAPACITY2)==0) rho_2 = atof(Values[i]);
			else if (strcmp(Names[i],P_THERMAL_CONDUCT0)==0) kappa_T0 = atof(Values[i]);
			else if (strcmp(Names[i],P_THERMAL_CONDUCT1)==0) kappa_T1 = atof(Values[i]);
			else if (strcmp(Names[i],P_THERMAL_CONDUCT2)==0) kappa_T2 = atof(Values[i]);
			else if (strcmp(Names[i],P_HEAT_FLUX_OUT)==0) kappa_bound = atof(Values[i]);
			else if (strcmp(Names[i],P_T_BATH)==0) T_outer = atof(Values[i]);
			else if (strcmp(Names[i],P_MARGIN_SIZE)==0) margin_large = atof(Values[i]);
			else if (strcmp(Names[i],P_MARGIN_TOUGHNESS)==0) large_toughness = atof(Values[i]);
			else if (strcmp(Names[i],P_ROBIN_STIFFNESS)==0) process_robin_arg(P_ROBIN_STIFFNESS,Values[i]);//Robin_stiffness = atof(Values[i]);

			else if (strcmp(Names[i],P_BACKTRACKING)==0) backtracking = get_bool(Values[i]);
			else if (strcmp(Names[i],P_ETOL)==0) energy_tol = atof(Values[i]);
			else if (strcmp(Names[i],P_TOTALTIME)==0) total_time = atof(Values[i]);
			else if (strcmp(Names[i],P_TIMESTEP)==0) dt = atof(Values[i]);
			else if (strcmp(Names[i],P_TIMESTEP_NUM)==0){
				char* endp;
				time_steps = strtol(Values[i],&endp,10);						
			}				
			else if (strcmp(Names[i],P_TIME_ADAPTIVE)==0) time_adaptive = get_bool(Values[i]);
			else if (strcmp(Names[i],P_TIME_ADAPTIVE_DZ)==0) time_adaptive_threshold_dH = atof(Values[i]);
			else if (strcmp(Names[i],P_TIME_MIN_DT)==0) time_adaptive_min_dt = atof(Values[i]);
			else if (strcmp(Names[i],P_CONSTRAINT_THRES)==0) eps_constraint = atof(Values[i]);
			
			else if (strcmp(Names[i],P_SPACE_ADAPTIVE)==0) space_adaptive = get_bool(Values[i]);
			else if (strcmp(Names[i],P_MIN_REFINEMENT)==0) min_refine_area = atof(Values[i]);
			else if (strcmp(Names[i],P_MAX_REFINEMENT)==0) max_refine_area = atof(Values[i]);
			
			else if (strcmp(Names[i],P_STRESS_FRACTION)==0) strain_ref_alpha_factor = atof(Values[i]);
			else if (strcmp(Names[i],P_STRESS_MAX_REFINE_AREA)==0) strain_ref_max_area = atof(Values[i]);
		
			//else if (strcmp(Names[i],P_MIN_EDGE_LEN)==0) min_edge_len = atof(Values[i]);
			//else if (strcmp(Names[i],P_EDGE_FACTOR)==0) edge_refine_fraction = atof(Values[i]);
			else if (strcmp(Names[i],P_INTERPOLATION_TOLERANCE)==0) interpolation_tol = atof(Values[i]);
			
			else if (strcmp(Names[i],P_ALT_EPS)==0) Alt_eps = atof(Values[i]);
			else if (strcmp(Names[i],P_ALT_MAXITER)==0) Alt_max_iter = atoi(Values[i]);
			else if (strcmp(Names[i],P_ALT_INFO)==0) Alt_info = get_bool(Values[i]);
			else if (strcmp(Names[i],P_ALT_TRUST_MIN)==0) Alt_trust_min = atof(Values[i]);
			else if (strcmp(Names[i],P_ALT_TRUST_MAX)==0) Alt_trust_max = atof(Values[i]);
			else if (strcmp(Names[i],P_ALT_TRUST_L)==0) Alt_trust_thres_min = atof(Values[i]);
			else if (strcmp(Names[i],P_ALT_TRUST_U)==0) Alt_trust_thres_max = atof(Values[i]);	
						
			//else if (strcmp(Names[i],P_CONSTRAINT_SOLVER)==0) constraint_solver = get_solver_number(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_EPS)==0) R_Newton_eps = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_MAXITER)==0) R_Newton_max_iter = atoi(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_SIGMA_L)==0) R_Newton_sigma_l = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_SIGMA_U)==0) R_Newton_sigma_u = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_RHO)==0) R_Newton_rho = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_DELTA)==0) R_Newton_Delta = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_TRUST_L)==0) R_Newton_trust_lower = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_TRUST_U)==0) R_Newton_trust_upper = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_TRUST_MIN)==0) R_Newton_trust_radius_min = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_TRUST_MAX)==0) R_Newton_trust_radius_max = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_SUB_TOL)==0) R_Newton_Dir_tol = atof(Values[i]);
			else if (strcmp(Names[i],P_R_NEWTON_INFO)==0) RN_print_info(get_bool(Values[i]));			
			//else if (strcmp(Names[i],P_BI_EPS)==0) BiCGStab_eps = atof(Values[i]);
			//else if (strcmp(Names[i],P_BI_MAXITER)==0) BiCGStab_max_iter = atoi(Values[i]);
			//else if (strcmp(Names[i],P_BI_INFO)==0) BI_print_info(get_bool(Values[i]));
			//else if (strcmp(Names[i],P_AMG_EPS)==0) step1_eps = atof(Values[i]);
			//else if (strcmp(Names[i],P_AMG_MAXITER)==0) step1_max_iter = atoi(Values[i]);
			//else if (strcmp(Names[i],P_AMG_SMOOTH_ITER)==0) AMG_smoothing_iter = atoi(Values[i]);
			//else if (strcmp(Names[i],P_AMG_DEPTH)==0) step1_AMG_depth = atoi(Values[i]);
			//else if (strcmp(Names[i],P_AMG_MATRIX_TOL)==0) step1_tol = atof(Values[i]);
			//else if (strcmp(Names[i],P_AMG_info)==0) AMG_print_info(get_bool(Values[i]));
	
			else printf("parameter %s not found -> ignore\n",Names[i]);			
		}
	}
	else{
		printf("Too few parameters in parameter file, found %d, required %d\n -> abort\n",len,P_COUNTER);
		exit(0);
	}
}

void Read_console_parameters(int argc,char* argv[]){
	const int Start = 3;
	int i,j,len;
	char** Line;
	int flag = 0;
	for (i=Start;i<argc;i++) if (argv[i][0]=='-'){
		Line = split(&(argv[i][1]),"=",&len);
		if (len>1){
			if (strcmp(Line[0],PS_OUTPUTNAME)==0) {sprintf(Output_name,"%s",Line[1]);printf("set %s: %s\n",P_OUTPUTNAME,Output_name);}
			else if (strcmp(Line[0],PS_OUTPUTDIR)==0) {sprintf(Output_dir,"%s",Line[1]);printf("set %s: %s\n",P_OUTPUTDIR,Output_dir);}
			else if (strcmp(Line[0],PS_MESHNAME)==0) {sprintf(Mesh_name,"%s",Line[1]);printf("set %s: %s\n",P_MESHNAME,Mesh_name);}
			else if (strcmp(Line[0],PS_SAMPLE_INTERVAL)==0) {sample_interval = atoi(Line[1]);printf("set %s: %d\n",P_SAMPLE_INTERVAL,sample_interval);}
			else if (strcmp(Line[0],PS_LOAD)==0) {sprintf(Load_name,"%s",Line[1]);printf("set %s: %s\n",P_LOAD,Load_name);}
			else if (strcmp(Line[0],PS_COND)==0) {sprintf(Cond_name,"%s",Line[1]);printf("set %s: %s\n",P_COND,Cond_name);}
			
			else if (strcmp(Line[0],PS_MU1)==0) {mu_1 = atof(Line[1]);flag=1;printf("set %s: %e\n",P_MU1,mu_1);}
			else if (strcmp(Line[0],PS_LAMBDA1)==0) {lambda_1 = atof(Line[1]);flag=1;printf("set %s: %e\n",P_LAMBDA1,lambda_1);}
			else if (strcmp(Line[0],PS_MU2)==0) {mu_2 = atof(Line[1]);flag=1;printf("set %s: %e\n",P_MU2,mu_2);}
			else if (strcmp(Line[0],PS_LAMBDA2)==0) {lambda_2 = atof(Line[1]);flag=1;printf("set %s: %e\n",P_LAMBDA2,lambda_2);}
			else if (strcmp(Line[0],PS_ALPHA1)==0) {a_1 = atof(Line[1]);printf("set %s: %e\n",P_ALPHA1,a_1);}
			else if (strcmp(Line[0],PS_ALPHA2)==0) {a_2 = atof(Line[1]);printf("set %s: %e\n",P_ALPHA2,a_2);}
			else if (strcmp(Line[0],PS_BETA)==0) {beta = atof(Line[1]);printf("set %s: %e\n",P_BETA,beta);}
			
			else if (strcmp(Line[0],PS_KAPPA1)==0){
				process_kappa_arg(P_KAPPA1,Line[1]);
				printf("set %s: %s\n",P_KAPPA1,Line[1]);
			}			
			else if (strcmp(Line[0],PS_KAPPA2)==0){
				process_kappa_arg(P_KAPPA2,Line[1]);
				printf("set %s: %s\n",P_KAPPA2,Line[1]);
			}
			
			else if (strcmp(Line[0],PS_XI)==0) {xi = atof(Line[1]);printf("set %s: %e\n",P_XI,xi);}
			else if (strcmp(Line[0],PS_ETA_EPS)==0) {eta_eps = atof(Line[1]);printf("set %s: %e\n",P_ETA_EPS,eta_eps);}
			else if (strcmp(Line[0],PS_MOBILITY1)==0) {mobility1 = atof(Line[1]);printf("set %s: %e\n",P_MOBILITY1,mobility1);}
			else if (strcmp(Line[0],PS_MOBILITY2)==0) {mobility2 = atof(Line[1]);printf("set %s: %e\n",P_MOBILITY2,mobility2);}
			else if (strcmp(Line[0],PS_KAPPA_C)==0) {kappa_c = atof(Line[1]);printf("set %s: %e\n",P_KAPPA_C,kappa_c);}
			else if (strcmp(Line[0],PS_EUTECTIC)==0) {c_eutectic = atof(Line[1]);printf("set %s: %e\n",P_EUTECTIC,c_eutectic);}
			else if (strcmp(Line[0],PS_ROBIN_STIFFNESS)==0) {
				char* S = strchr(&(argv[i][1]),'=');
				process_robin_arg(P_ROBIN_STIFFNESS,++S);
				printf("set %s: %s\n",P_ROBIN_STIFFNESS,++S);
			}
			//{Robin_stiffness = atof(Line[1]);printf("set %s: %e\n",P_ROBIN_STIFFNESS,Robin_stiffness);}
			
			else if (strcmp(Line[0],PS_THERMAL_CONDUCT1)==0) {kappa_T1= atof(Line[1]);printf("set %s: %e\n",PS_THERMAL_CONDUCT1,kappa_T1);}
			else if (strcmp(Line[0],PS_THERMAL_CONDUCT2)==0) {kappa_T2= atof(Line[1]);printf("set %s: %e\n",PS_THERMAL_CONDUCT2,kappa_T2);}
			else if (strcmp(Line[0],PS_THERMAL_CONDUCT0)==0) {kappa_T0= atof(Line[1]);printf("set %s: %e\n",PS_THERMAL_CONDUCT0,kappa_T0);}
			
			else if (strcmp(Line[0],PS_THERMAL_EXPANSION1)==0) {
				process_thermexp_arg(P_THERMAL_EXPANSION1,Line[1]);
				printf("set %s: %s\n",P_THERMAL_EXPANSION1,Line[1]);
			}
			else if (strcmp(Line[0],PS_THERMAL_EXPANSION2)==0) {
				process_thermexp_arg(P_THERMAL_EXPANSION2,Line[1]);
				printf("set %s: %s\n",P_THERMAL_EXPANSION2,Line[1]);
			}
			
			else if (strcmp(Line[0],PS_MODULI1)==0) {				
				char* S = strchr(&(argv[i][1]),'=');
				process_elastic_moduli_arg(P_MU1,++S);
				printf("set %s: %s\n",PS_MODULI1,++S);
			}
			else if (strcmp(Line[0],PS_MODULI2)==0) {
				char* S = strchr(&(argv[i][1]),'=');
				process_elastic_moduli_arg(P_MU2,++S);
				printf("set %s: %s\n",PS_MODULI2,++S);
			}
			
			else if (strcmp(Line[0],PS_ELEMENT_INTERPOLATION)==0) {				
				if(strcmp(Line[1],"yes")==0) interpolation_method = ELEMENT_BASE;
				if(strcmp(Line[1],"no")==0) interpolation_method = BARYCENTRIC;
				printf("set element interpolation: %s\n",Line[1]);
			}
			
			else if (strcmp(Line[0],PS_BACKTRACKING)==0) {backtracking = get_bool(Line[1]);printf("set %s: %s\n",P_BACKTRACKING,get_string_bool(backtracking));}
			else if (strcmp(Line[0],PS_TOTALTIME)==0) {total_time = atof(Line[1]);printf("set %s: %e\n",P_TOTALTIME,total_time);}
			else if (strcmp(Line[0],PS_TIMESTEP)==0) {dt = atof(Line[1]);printf("set %s: %e\n",P_TIMESTEP,dt);}
			else if (strcmp(Line[0],PS_TIMESTEP_NUM)==0) {
				char* endp;
				time_steps = strtol(Line[1],&endp,10);
				printf("set %s: %ld\n",P_TIMESTEP_NUM,time_steps);				
			}
			else if (strcmp(Line[0],PS_TIME_ADAPTIVE)==0) {time_adaptive = get_bool(Line[1]);printf("set %s: %s\n",P_TIME_ADAPTIVE,get_string_bool(time_adaptive));}
			else if (strcmp(Line[0],PS_SPACE_ADAPTIVE)==0) {space_adaptive = get_bool(Line[1]);printf("set %s: %s\n",P_SPACE_ADAPTIVE,get_string_bool(space_adaptive));}
			else if (strcmp(Line[0],PS_REFINE_AMIN)==0) {min_refine_area = atof(Line[1]);printf("set %s: %e\n",PS_REFINE_AMIN,min_refine_area);}
			else if (strcmp(Line[0],PS_TIME_MIN_DT)==0) {time_adaptive_min_dt = atof(Line[1]);printf("set %s: %e\n",PS_TIME_MIN_DT,time_adaptive_min_dt);}						
			
			else if (strcmp(Line[0],PS_ALT_INFO)==0) {Alt_info = get_bool(Line[1]);printf("set %s: %s\n",P_ALT_INFO,get_string_bool(Alt_info));}
			
			else if (strcmp(Line[0],OMP_PROC_NUM)==0) {omp_proc_num = atoi(Line[1]);printf("set %s: %d\n","number of omp processes\n",omp_proc_num);}
			
			else if (strcmp(Line[0],LOAD_INDEX)==0) {load_index = atoi(Line[1]);printf("load time index %d\n",load_index);}
			
			else if (strcmp(Line[0],PS_TIME_OFFSET)==0) {time_offset = atof(Line[1]);printf("set time offset %f\n",time_offset);}
			
			else if (strcmp(Line[0],PS_CONSTRAINT_THRES)==0) {eps_constraint = atof(Line[1]);printf("set %s: %e\n",P_CONSTRAINT_THRES,eps_constraint);}
			
			else if (strcmp(Line[0],ES_INT_COEFF)==0){a_interface = atof(Line[1]);printf("set interface toughness slope: %e\n",a_interface);}
			else if (strcmp(Line[0],ES_INT_ALPHA)==0){interface_thougness_min = atof(Line[1]);printf("set interface toughness reduction: %e\n",interface_thougness_min);}
		}
		else printf("invalid parameter input: %s -> ignore\n",argv[i]);
		for (j=0;j<len;j++) free(Line[j]);
		free(Line);
	}
	
	// update elastic moduli of changed by console
	if (flag){
		Moduli_1.xxxx = 2.*mu_1+lambda_1;
		Moduli_1.yyyy = 2.*mu_1+lambda_1;
		Moduli_1.xxxy = 0;
		Moduli_1.yyyx = 0;
		Moduli_1.xxyy = lambda_1;
		Moduli_1.xyxy = mu_1;
		
		Moduli_2.xxxx = 2.*mu_2+lambda_2;
		Moduli_2.yyyy = 2.*mu_2+lambda_2;
		Moduli_2.xxxy = 0;
		Moduli_2.yyyx = 0;
		Moduli_2.xxyy = lambda_2;
		Moduli_2.xyxy = mu_2;		
	}
}

void Write_parameters(const char* Filename){
	char Name[512];
	char Defines[1024];
	time_t t = time(NULL);
	struct tm date = *localtime(&t);
	
	Defines[0] = '\0';
	#ifdef ANISOTROPIC_ELASTICITY
		strcat(Defines,"anisotropic,");
	#else
		strcat(Defines,"isotropic,");
	#endif
	#ifdef NO_MARGIN
		strcat(Defines,"no margin,");
	#endif
	#ifdef X_MARGIN
		strcat(Defines,"x margin,");
	#endif
	#ifdef X_STRETCH
		strcat(Defines,"x stretch,");
	#endif
	#ifdef Y_STRETCH
		strcat(Defines,"y-stretch,");
	#endif
	
	switch(strain_decomposition){
		case NO: strcat(Defines,"stretch-compression anisotropy: none,");break;
		case TRACE: strcat(Defines,"stretch-compression anisotropy: shear/trace,");break;
		case EIGENSYSTEM: strcat(Defines,"stretch-compression anisotropy: none,");break;
	}
	
	if (interpolation_method==BARYCENTRIC) strcat(Defines,"interpolation: barycentric,");
	else strcat(Defines,"interpolation: element base,");
	
	sprintf(Name,"damage zero-slope: %f, ",damage_zero_slope);
	strcat(Defines,Name);
	
	sprintf(Name,"%s/%s/%s",Output_dir,Output_name,Filename);
	FILE* file = fopen(Name,"w");
	if (file!=NULL){
		fprintf(file,"%s\t%d-%d-%d\t%s\n",Prog_name,date.tm_mon+1,date.tm_mday,date.tm_year+1900,Defines);
		
		fprintf(file,"%s\t%s\n",P_OUTPUTNAME,Output_name);
		fprintf(file,"%s\t%s\n",P_OUTPUTDIR,Output_dir);
		fprintf(file,"%s\t%s\n",P_MESHNAME,Mesh_name);
		fprintf(file,"%s\t%s\n",P_MESHDIR,Mesh_dir);	
		fprintf(file,"%s\t%s\n",P_COND,Cond_name);	
		fprintf(file,"%s\t%d\n",P_SAMPLE_INTERVAL,sample_interval);		
		fprintf(file,"%s\t%s\n",P_LOAD,Load_name);
	
	#ifndef ANISOTROPIC_ELASTICITY
		fprintf(file,"%s\t%e\n",P_MU1,mu_1);
		fprintf(file,"%s\t%e\n",P_MU2,mu_2);
	#else 
		print_moduli(file,P_MU1);
		print_moduli(file,P_MU2);
	#endif
		fprintf(file,"%s\t%e\n",P_LAMBDA1,lambda_1);
		fprintf(file,"%s\t%e\n",P_LAMBDA2,lambda_2);		
		fprintf(file,"%s\t%e\n",P_ALPHA1,a_1);
		fprintf(file,"%s\t%e\n",P_ALPHA2,a_2);
		fprintf(file,"%s\t%e\n",P_BETA,beta);
		print_kappa(file,P_KAPPA1);
		print_kappa(file,P_KAPPA2);			
		fprintf(file,"%s\t%e\n",P_XI,xi);
		fprintf(file,"%s\t%e\n",P_ETA_EPS,eta_eps);
		fprintf(file,"%s\t%e\n",P_MOBILITY1,mobility1);
		fprintf(file,"%s\t%e\n",P_MOBILITY2,mobility2);
		fprintf(file,"%s\t%e\n",P_KAPPA_C,kappa_c);
		fprintf(file,"%s\t%e\n",P_LANDAU_POT,pot_factor);
		fprintf(file,"%s\t%e\n",P_VISCOSITY_C,vis_c);
		fprintf(file,"%s\t%e\n",P_T_CRITICAL,T_crit);
		fprintf(file,"%s\t%e\n",P_EUTECTIC,c_eutectic);
		fprintf(file,"%s\t%e\n",P_T_INITIAL,T_ref);
		print_thermexp(file,P_THERMAL_EXPANSION1);
		print_thermexp(file,P_THERMAL_EXPANSION2);		
		fprintf(file,"%s\t%e\n",P_HEAT_CAPACITY0,rho_0);
		fprintf(file,"%s\t%e\n",P_HEAT_CAPACITY1,rho_1);
		fprintf(file,"%s\t%e\n",P_HEAT_CAPACITY2,rho_2);
		fprintf(file,"%s\t%e\n",P_THERMAL_CONDUCT0,kappa_T0);
		fprintf(file,"%s\t%e\n",P_THERMAL_CONDUCT1,kappa_T1);
		fprintf(file,"%s\t%e\n",P_THERMAL_CONDUCT2,kappa_T2);
		fprintf(file,"%s\t%e\n",P_HEAT_FLUX_OUT,kappa_bound);
		fprintf(file,"%s\t%e\n",P_T_BATH,T_outer);
		
		fprintf(file,"%s\t%s\n",P_BACKTRACKING,get_string_bool(backtracking));
		fprintf(file,"%s\t%e\n",P_ETOL,energy_tol);
		fprintf(file,"%s\t%e\n",P_TOTALTIME,total_time);
		fprintf(file,"%s\t%e\n",P_TIMESTEP,dt);
		fprintf(file,"%s\t%ld\n",P_TIMESTEP_NUM,time_steps);
		fprintf(file,"%s\t%s\n",P_TIME_ADAPTIVE,get_string_bool(time_adaptive));
		fprintf(file,"%s\t%e\n",P_TIME_ADAPTIVE_DZ,time_adaptive_threshold_dH);
		fprintf(file,"%s\t%e\n",P_TIME_MIN_DT,time_adaptive_min_dt);
		fprintf(file,"%s\t%e\n",P_CONSTRAINT_THRES,eps_constraint);
		fprintf(file,"%s\t%s\n",P_SPACE_ADAPTIVE,get_string_bool(space_adaptive));
		
		fprintf(file,"%s\t%e\n",P_MIN_REFINEMENT,min_refine_area);
		fprintf(file,"%s\t%e\n",P_STRESS_FRACTION,strain_ref_alpha_factor);
		fprintf(file,"%s\t%e\n",P_STRESS_MAX_REFINE_AREA,strain_ref_max_area);

		//fprintf(file,"%s\t%e\n",P_MAX_REFINEMENT,max_refine_area);
		//fprintf(file,"%s\t%e\n",P_MIN_EDGE_LEN,min_edge_len);
		//fprintf(file,"%s\t%e\n",P_EDGE_FACTOR,edge_refine_fraction);		
		fprintf(file,"%s\t%e\n",P_INTERPOLATION_TOLERANCE,interpolation_tol);
		
		fprintf(file,"%s\t%e\n",P_ALT_EPS,Alt_eps);		
		fprintf(file,"%s\t%d\n",P_ALT_MAXITER,Alt_max_iter);
		fprintf(file,"%s\t%s\n",P_ALT_INFO,get_string_bool(Alt_info));
		fprintf(file,"%s\t%e\n",P_ALT_TRUST_MIN,Alt_trust_min);
		fprintf(file,"%s\t%e\n",P_ALT_TRUST_MAX,Alt_trust_max);
		fprintf(file,"%s\t%e\n",P_ALT_TRUST_L,Alt_trust_thres_min);
		fprintf(file,"%s\t%e\n",P_ALT_TRUST_U,Alt_trust_thres_max);		
		
		//fprintf(file,"%s\t%s\n",P_CONSTRAINT_SOLVER,get_solver_string(constraint_solver));
		fprintf(file,"%s\t%e\n",P_R_NEWTON_EPS,R_Newton_eps);
		fprintf(file,"%s\t%d\n",P_R_NEWTON_MAXITER,R_Newton_max_iter);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_SIGMA_L,R_Newton_sigma_l);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_SIGMA_U,R_Newton_sigma_u);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_RHO,R_Newton_rho);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_DELTA,R_Newton_Delta);		
		fprintf(file,"%s\t%e\n",P_R_NEWTON_TRUST_L,R_Newton_trust_lower);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_TRUST_U,R_Newton_trust_upper);
		
		fprintf(file,"%s\t%e\n",P_R_NEWTON_TRUST_MIN,R_Newton_trust_radius_min);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_TRUST_MAX,R_Newton_trust_radius_max);
		fprintf(file,"%s\t%e\n",P_R_NEWTON_SUB_TOL,R_Newton_Dir_tol);
		fprintf(file,"%s\t%s\n",P_R_NEWTON_INFO,get_string_bool(RN_info_set()));
		//fprintf(file,"%s\t%e\n",P_BI_EPS,BiCGStab_eps);		
		//fprintf(file,"%s\t%d\n",P_BI_MAXITER,BiCGStab_max_iter);
		//fprintf(file,"%s\t%s\n",P_BI_INFO,get_string_bool(BI_info_set()));
		
		//fprintf(file,"%s\t%e\n",P_AMG_EPS,step1_eps);
		//fprintf(file,"%s\t%d\n",P_AMG_MAXITER,step1_max_iter);
		//fprintf(file,"%s\t%d\n",P_AMG_SMOOTH_ITER,AMG_smoothing_iter);
		//fprintf(file,"%s\t%d\n",P_AMG_DEPTH,step1_AMG_depth);		
		//fprintf(file,"%s\t%e\n",P_AMG_MATRIX_TOL,step1_tol);
		//fprintf(file,"%s\t%s\n",P_AMG_info,get_string_bool(AMG_info_set()));
		
		fprintf(file,"%s\t%e\n",P_MARGIN_SIZE,margin_large);
		fprintf(file,"%s\t%e\n",P_MARGIN_TOUGHNESS,large_toughness);
		print_robin(file,P_ROBIN_STIFFNESS);
		//fprintf(file,"%s\t%e\n",P_ROBIN_STIFFNESS,Robin_stiffness);
		
		fclose(file);
	}
	else printf("could not write to file %s -> skip\n",Name);	
}

FILE* open_global(){
	char Fullname[512];
	char Data[512];
	sprintf(Fullname,"%s/%s",Output_dir,Output_name);
	sprintf(Data,"%s/%s",Fullname,"global");
	return fopen(Data,"a");
}

FILE* open_local(){
	char Fullname[512];
	char Data[512];
	sprintf(Fullname,"%s/%s",Output_dir,Output_name);
	sprintf(Data,"%s/%s",Fullname,"local");
	return fopen(Data,"a");
}

double ln_abs(double x){
	return log(fabs(x));
}

void save(double* Sol,double* Globals,int glob_size,int t_index){
	const int SIZE = 512;

	static int counter = 0;
	
	int i,j;
	int n = glob_mesh.size;
	char Buffer[SIZE];
	char Name[SIZE];	
	
	// global output	
	FILE* Output_global = open_global();
	if (Globals!=NULL && Output_global!=NULL){
		char* Line = (char*)malloc(SIZE*glob_size*sizeof(char));
		sprintf(Line,"%.10f\t",Times[t_index]);
		for (i=0;i<glob_size-1;i++){
			sprintf(Buffer,"%e\t",Globals[i]);
			strcat(Line,Buffer);
		}
		sprintf(Buffer,"%e\n",Globals[glob_size-1]);
		strcat(Line,Buffer);
		if (fprintf(Output_global,"%s",Line)<0){
			printf("error writing output file -> abort\n");
			exit(0); 
		}
		free(Line);
	}
	if (Output_global!=NULL){
		fflush(Output_global);
		fclose(Output_global);
	}
	
	//local output
	if (Sol!=NULL){
		double* H_prev;
		double* C_prev;		
		double* All_fields;
	
		double* U = &(Sol[0]); 
		double* H = &(Sol[dimension*n]); 
		double* C = &(Sol[(dimension+1)*n]);
		double* Y = &(Sol[(dimension+2)*n]);
		double* T = &(Sol[(dimension+3)*n]);
		
		double* V = set_vector_Ui_0_2D(1);
		if (t_index>0) H_prev = interpolated(All_H[t_index-1],1,Glob_mesh_sizes[t_index-1],NULL,NULL); else H_prev = clone_vector(All_H[t_index],n);
		
		// additional fields
		
		double* AF = zero_vector(add_field_num*n);
		
		Strain_energy = elastic_potential(U,H,C,T,FACTOR,PARTIAL0);		
		double* Strain_energy_self = elastic_potential_self(U,H,C,T,FACTOR,PARTIAL0,PARTIAL0);
		vector_add(Strain_energy,Strain_energy_self,1.,n);	
		free(Strain_energy_self);
		scalar_mult(2.,Strain_energy,n);
		double* dH = damage_driving_force(H,H_prev,C);
		copy_vector_content(dH,AF,0,n,n);	
		free(dH);
		/*double* Tr = filtered_total_strain_trace(U,H,C,T);
		copy_vector_content(Tr,AF,0,0,n);	
		free(Tr);*/
		
		
		/*double* G = damage_weight_function_dH(H);
		vector_pseudo_mult(G,Strain_energy,n);
		vector_pseudo_div(G,V,n);
		copy_vector_content(G,AF,0,n,n);		
		free(G);*/
		
		free(Strain_energy);
		//free(dH);
		
		
		double* S = compute_strain_energy(U,H,C,T,PARTIAL0);
		copy_vector_content(S,AF,0,0,n);		
		free(S);
		
		//double* Det = strain_det(U);
		//copy_vector_content(Det,AF,0,n,n);		
		//free(Det);
				
		double* Eig = zero_vector(2*n);
		double* Vec1 = zero_vector(2*n);
		double* Vec2 = zero_vector(2*n);
		total_strain_eigensystem(U,C,T,Vec1,Vec2,Eig);
		copy_vector_content(Eig,AF,0,2*n,2*n);
		vector_add(&(AF[2*n]),&(Eig[n]),-1.,n);								//e1-e2
		vector_add(&(AF[3*n]),&(Eig[0]),1.,n);								//e1+e2
		copy_vector_content(Vec1,AF,0,4*n,2*n);
		free(Eig);
		free(Vec1);
		free(Vec2);
		
		double* Refine = zero_vector(n);
		refine_by_gradient(Sol,Refine,degrees_of_freedom,0,max_refine_area);
		copy_vector_content(Refine,AF,0,2*n,n);
		free(Refine);
		
		//double* Rank4_modulus = elastic_constitutive(U,H,C,T,PARTIAL0,PARTIAL0);
		//copy_vector_content(Rank4_modulus,AF,0,n,n);
		//free(Rank4_modulus);		
	
		//double* AF = max_steepnes_2D(Sol,degrees_of_freedom);
		
		//double* AF = get_constraint_force(U,H,H_prev,C,T);	
		
		/*double mean = mean2D(AF);
		double var = moment_2D(AF,10);
		double extr = get_max_element_abs(AF,n);
		printf("\nlarge stress extrema: mean=%e var=%e max=%e\n",mean,var,extr);*/
		
		//double* AF = Heat_source(H,H_prev,Y);
		//vector_pseudo_div(AF,V,n);				
		
		// join all together		
		All_fields = join_vectors(Sol,degrees_of_freedom*n,AF,add_field_num*n);
		
		// save
		sprintf(Name,"%s/%s/%s",Output_dir,Output_name,Bin_filename);
		binary_save(Name,All_fields,(degrees_of_freedom+add_field_num)*n,Times[t_index],glob_mesh_index);
		
		/*search_tree2D* Tree = create_search_tree2D(&glob_mesh,&elements);
		point2D p = init_point2D(0.3,0.2);
		int ind = get_triangle_index(Tree,&p);
		printf("tringle index: %d\n",ind);
		free_search_tree2D(&Tree);
		exit(0);*/
		
		// clean
		free(H_prev);
		free(AF);
		free(V);
		free(All_fields);
		
		counter++;
	}
}

int init_binary_save(const char* Name){
	int i;
	char Fullname[512];
	int n = degrees_of_freedom+add_field_num;
	int m = 0;
	sprintf(Fullname,"%s/%s/%s",Output_dir,Output_name,Name);
	FILE* Dat = fopen(Fullname,"w");
	if (Dat!=NULL){
		fwrite(&n,sizeof(int),1,Dat);
		fwrite(&m,sizeof(int),1,Dat);
		fclose(Dat);
		
		char* FNames[] = FIELD_NAMES;
		sprintf(Fullname,"%s/%s/%s.info",Output_dir,Output_name,Name);
		Dat = fopen(Fullname,"w");
		fprintf(Dat,"%d\n",degrees_of_freedom+add_field_num);
		for (i=0;i<degrees_of_freedom;i++) fprintf(Dat,"%s\n",FNames[i]);
		for (i=0;i<add_field_num;i++) fprintf(Dat,"%s %d\n",F_ADD,i+1);
		
		fclose(Dat);
		
		return SUCCESS;
	}
	else{
		printf("Warning: could not create binary file: %s\n",Fullname);
		return FAILED;
	}
}

int binary_save(char* Fullname,double* X,int size,double t,int mesh_index){
	//char Fullname[512];
	int i,n,m;
	int N = glob_mesh.size;
	//int N = degrees_of_freedom*glob_mesh.size;
	double time = t;
	//sprintf(Fullname,"%s/%s/%s",Output_dir,Output_name,Name);
	FILE* Dat = fopen(Fullname,"r+w");
	
	
	
	/*int error = 0;
	for (i=0;i<size;i++) if (isnan(X[i])){
		printf("Warning binary_save: NAN encountered at index %d\n",i);
		error = 1;
	}*/
	
	if (Dat!=NULL){
		
		// read and change header
		fread(&n,sizeof(int),1,Dat);
		fread(&m,sizeof(int),1,Dat);
		/*if ((n % N)!=0){
			printf("binary save: Wrong mesh size %d-> skip\n",n);
			return FAILED;
		}*/
		m++;
		n = degrees_of_freedom+add_field_num;
		//fseek(Dat,-sizeof(int),SEEK_CUR);
		fseek(Dat,0,SEEK_SET);
		fwrite(&n,sizeof(int),1,Dat);
		fwrite(&m,sizeof(int),1,Dat);
		
		
		// write data
		fseek(Dat,0,SEEK_END);
		fwrite(&time,sizeof(double),1,Dat);
		fwrite(&mesh_index,sizeof(int),1,Dat);
		fwrite(X,sizeof(double),size,Dat);
		
		// close
		fclose(Dat);
	}
	else{
		printf("Warning: could not write to file %s-> skip\n",Fullname);
		return FAILED;
	}
	return SUCCESS;
}

int load_mesh_sizes(char* Loadname,int** Sizes,int* len){
	const int BUFF_SIZE = 1024;
	
	char** Parts;
	char Name[BUFF_SIZE];
	char Buffer[BUFF_SIZE];
	char Basename[BUFF_SIZE];
	char Param_name[BUFF_SIZE];
	int i,plen,flag;
	
	int index = 0;
	FILE* Dat = NULL;
	
	*Sizes = NULL;
	*len = 0;
	
	sprintf(Param_name,"%s/%s",Loadname,Default_param_name);
	char* Meshname = Get_parameter(Param_name,P_MESHNAME);
	
	Parts = split(Meshname,".",&plen);
	sprintf(Basename,"%s",Parts[0]);
	for (i=0;i<plen;i++) free(Parts[i]);
	free(Parts);
	
	do{
		flag = 0;
		index++;
		sprintf(Name,"%s/meshes/%s.%d.node",Loadname,Basename,index);
		FILE* Dat = fopen(Name,"r");
		if (Dat!=NULL){
			flag = 1;
			if(fgets(Buffer,BUFF_SIZE,Dat)!=NULL){
				plen = 0;
				Parts = split(Buffer," ",&plen);
				if (plen>0){
					(*len)++;
					*Sizes = (int*)realloc(*Sizes,(*len)*sizeof(int));
					(*Sizes)[(*len)-1] = atoi(Parts[0]);
				}
				for (i=0;i<plen;i++) free(Parts[i]);
				free(Parts);
				//printf("index: %d size: %d\n",index,(*Sizes)[(*len)-1]);
			}
			fclose(Dat);
		}
	}while(flag==1);
	
	free(Meshname);
	if (*len>0){
		printf("load from file: %d meshes found\n",*len);
		return SUCCESS;
	}
	else{
		printf("Warning: could not find meshes %s",Name);
		return FAILED;
	}
}

int binary_load(char* Fullname,double** X,double* time,int* size,int index){				
	int i,n,m,r,mesh_index;																 	
	char Datname[1024];
	
	int* Mesh_sizes = NULL;
	int mesh_number = 0;
	if (load_mesh_sizes(Fullname,&Mesh_sizes,&mesh_number)==FAILED) return FAILED;
	
	sprintf(Datname,"%s/%s",Fullname,Bin_filename);
	FILE* Dat = fopen(Datname,"r");
	if (Dat!=NULL){
		
		fread(&n,sizeof(int),1,Dat);
		fread(&m,sizeof(int),1,Dat);
		if (index==LAST_INDEX) index = m-1;
		if (index==PREV_LAST_INDEX) index = m-2;
		
		if (index<0 || index>=m){
			printf("binary load: Index %d out of range 0-%d -> skip\n",index,m);
			return FAILED;
		}
		
		// goto index
		for (i=0;i<index;i++){
			fread(time,sizeof(double),1,Dat);
			fread(&mesh_index,sizeof(int),1,Dat);
			if (mesh_index>mesh_number){
				printf("error in binary load: mesh index %d exceeds number of meshes %d\n -> abort\n",mesh_index,mesh_number);
				exit(0);
			}
			fseek(Dat,n*Mesh_sizes[mesh_index-1]*sizeof(double),SEEK_CUR);
		}
		
		// read time
		fread(time,sizeof(double),1,Dat);
		fread(&mesh_index,sizeof(int),1,Dat);
		
		// read data
		*size = n*Mesh_sizes[mesh_index-1];
		*X = (double*)malloc((*size)*sizeof(double));
		r = fread(*X,sizeof(double),*size,Dat);
		if (n!=0 && r==0){
			printf("error reading file %s -> skip\n",Datname);
			return FAILED;
		}
		glob_mesh_index = mesh_index;
		
		// close
		fclose(Dat);
	}
	else{
		printf("Warning: could not open file %s-> skip\n",Datname);
		return FAILED;
	}
	return SUCCESS;	
}

	
double* compute_divergence(double* X){	
	const int max_iter = 100;
	const double eps = 1e-10;
	
	int m = get_var_number2D();
	int n = glob_mesh.size;
	set_var_number2D(1);
	sparse_matrix* ID = set_matrix_Aij_00_2D(0,0,&insert_AV);
	set_var_number2D(2);
	sparse_matrix* Div = set_matrix_Aij_01_2D(0,0,&insert_A_aV_a);
	double* Div2 = sparse_mult(Div,X);
	double* Div1 = restrict_vector_by(0,n,Div2);
	sparse_matrix* L = sparse_zero(n);
	sparse_matrix* U = sparse_zero(n);
	incomplete_LU_factorization(ID,L,U);
	PCG_default(ID,L,U);
	double* Res = zero_vector(n);
	
	double r0 = matrix_residuum(ID,Res,Div1);
	if (r0>0){
		if (r0>0) pcg(Res,Div1,n,eps,max_iter);
		double r = matrix_residuum(ID,Res,Div1);		
		//printf("residuum divergence: %e\n",r/r0);
	}
	
	set_var_number2D(m);
	free_sparse(ID);
	free_sparse(Div);
	free_sparse(L);
	free_sparse(U);
	free(Div1);
	free(Div2);
	return Res;
}

double* max_steepnes_2D(double* X,double* Weight,int field_number){
	const double eps = 1e-8;
	
	int i,j;
	double g,d,min,max,mean,variance;
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	
	set_var_number2D(2);
	double* V = set_vector_Ui_0_2D(2);
	sparse_matrix* Grad = set_matrix_Aij_01_2D(0,0,&insert_A_aV);
	for (i=0;i<field_number;i++){
		
		min = X[i*n];
		max = min;
		for (j=1;j<n;j++){
			if (X[i*n+j]<min) min = X[i*n+j];
			if (X[i*n+j]>max) max = X[i*n+j];
		}
		d = max-min;
		
		if (d>eps){
			double* G = join_vectors(&(X[i*n]),n,&(X[i*n]),n);
			sparse_multiplication(Grad,G);
			for (j=0;j<n;j++){
				g = Weight[i]*sqrt(G[j]*G[j]+G[n+j]*G[n+j])/(V[j]*d);
				if (g>Res[j]) Res[j] = g;
			}
			free(G);
		}
	}
	free_sparse(Grad);
	free(V);
	return Res;
}

int multifrontal_solver_get_LU(sparse_matrix* A,void** Factor_info,umfpack_matrix_info* Matrix_info){
	const int system_type = UMFPACK_A; 														 // default: solves Ax=b
	void* Pattern_info = NULL;
	int status;
	int n = A->size;
	int el_number = 0;
	
	convert_sparse_to_UMFPACK(A,n,&(Matrix_info->Col),&(Matrix_info->Ind),&(Matrix_info->Val),&el_number);
	status = umfpack_di_symbolic(n,n,Matrix_info->Col,Matrix_info->Ind,Matrix_info->Val,&Pattern_info,
	  UMFPACK_pattern_options,UMFPACK_pattern_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_ERROR_invalid_matrix: printf("UMFPACK symbolic: invalid matrix\n");break;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK symbolic: memory insufficient\n");break;
			case UMFPACK_ERROR_internal_error: printf("UMFPACK symbolic: internal error\n");break;
			default: printf("UMFPACK symbolic: unknown error\n");break;
		}
		exit(0);
	}
	
	status = umfpack_di_numeric(Matrix_info->Col,Matrix_info->Ind,Matrix_info->Val,Pattern_info,Factor_info,
	  UMFPACK_factor_options,UMFPACK_factor_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK numeric: singular matrix\n");
				umfpack_di_free_symbolic(&Pattern_info);	
				return FAILED;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK numeric: memory insufficient\n");break;
			case UMFPACK_ERROR_invalid_Symbolic_object: printf("UMFPACK numeric: pattern info invalid\n");break;	
			case UMFPACK_ERROR_different_pattern: printf("UMFPACK numeric: pattern info has changed\n");break;	
			default: printf("UMFPACK numeric: unknown error\n");break;		
		}
		exit(0);
	}
	if (UMFPACK_factor_info[UMFPACK_RCOND]>max_cond_num) max_cond_num = UMFPACK_factor_info[UMFPACK_RCOND];
	if (UMFPACK_factor_info[UMFPACK_RCOND]<min_cond_num) min_cond_num = UMFPACK_factor_info[UMFPACK_RCOND];
	
	umfpack_di_free_symbolic(&Pattern_info);
	return SUCCESS;
}

int multifrontal_solver_solve_LU(void* Factor_info,umfpack_matrix_info* Matrix_info,double* X,double* b){
	const int system_type = UMFPACK_A; 														 // default: solves Ax=b
	
	int status;
	status = umfpack_di_solve(system_type,Matrix_info->Col,Matrix_info->Ind,Matrix_info->Val,X,b,Factor_info,
	  UMFPACK_solve_options,UMFPACK_solve_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK solve: singular matrix\n");
				return FAILED;				
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK solve: memory insufficient\n");break;
			case UMFPACK_ERROR_invalid_Numeric_object: printf("UMFPACK solve: factor info invalid\n");break;	
			case UMFPACK_ERROR_invalid_system: printf("UMFPACK solve: invalid system\n");break;		
			default: printf("UMFPACK solve: unknown error\n");break;	
		}
		exit(0);
	}	
	return SUCCESS;
}

int multifrontal_solver(sparse_matrix* A,double* X,double* b,char* Caller,double* time){
	const int system_type = UMFPACK_A; 														 // default: solves Ax=b
	
	static double total_umfpack_time = 0;
	
	double Time[2];
	if (A==NULL) return round(total_umfpack_time);
	
	double Control [UMFPACK_CONTROL];
	umfpack_di_defaults (Control) ;
	//Control[UMFPACK_PIVOT_TOLERANCE] = 1e-8;
	//Control [UMFPACK_SYM_PIVOT_TOLERANCE] = 1e-6;

	
	static void* Pattern_info = NULL;
	static void* Factor_info = NULL;
	
	int* A_col = NULL;
	int* A_ind = NULL;
	double* A_val = NULL;	
	int el_number = 0;
	
	int status;
	int n = A->size;
	
	umfpack_tic(Time);
	
	convert_sparse_to_UMFPACK(A,n,&A_col,&A_ind,&A_val,&el_number);
	status = umfpack_di_symbolic(n,n,A_col,A_ind,A_val,&Pattern_info,Control,UMFPACK_pattern_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_ERROR_invalid_matrix: printf("UMFPACK symbolic: invalid matrix in %s\n",Caller);break;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK symbolic: memory insufficient in %s\n",Caller);break;
			case UMFPACK_ERROR_internal_error: printf("UMFPACK symbolic: internal error in %s\n",Caller);break;
			default: printf("UMFPACK symbolic: unknown error\n");break;
		}
		exit(0);
	}
	
	status = umfpack_di_numeric(A_col,A_ind,A_val,Pattern_info,&Factor_info,Control,UMFPACK_factor_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK numeric: singular matrix in %s\n",Caller);				
				umfpack_di_free_symbolic(&Pattern_info);					
				free(A_col);
				free(A_ind);
				free(A_val);
				print_sparse(A);
				print_to_matrix_market_format(A,"/Home/damage/radszuwe/local/FILTLAN/DATA/singalur_matrix.mtx");
				print_vector(X,n);
				break;
				//return FAILED;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK numeric: memory insufficient in %s\n",Caller);break;
			case UMFPACK_ERROR_invalid_Symbolic_object: printf("UMFPACK numeric: pattern info invalid in %s\n",Caller);break;	
			case UMFPACK_ERROR_different_pattern: printf("UMFPACK numeric: pattern info has changed in %s\n",Caller);break;			
			default: printf("UMFPACK symbolic: numeric error\n");break;
		}
		
		int j;
		void* Addr_buffer[1024];
		char** Func_names = NULL;
		int f = backtrace(Addr_buffer,1024);
		Func_names = backtrace_symbols(Addr_buffer,f);
		printf("\nin:\n");
		if (Func_names!=NULL){
			for (j=0;j<f;j++) printf("%s\n",Func_names[j]);
		}
		
		exit(0);
	}
	if (UMFPACK_factor_info[UMFPACK_RCOND]>max_cond_num) max_cond_num = UMFPACK_factor_info[UMFPACK_RCOND];
	if (UMFPACK_factor_info[UMFPACK_RCOND]<min_cond_num) min_cond_num = UMFPACK_factor_info[UMFPACK_RCOND];
	
	status = umfpack_di_solve(system_type,A_col,A_ind,A_val,X,b,Factor_info,Control,UMFPACK_solve_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK solve: singular matrix\n");
				umfpack_di_free_symbolic(&Pattern_info);
				umfpack_di_free_numeric(&Factor_info);
				free(A_col);
				free(A_ind);
				free(A_val);
				return FAILED;				
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK solve: memory insufficient in %s\n",Caller);break;
			case UMFPACK_ERROR_invalid_Numeric_object: printf("UMFPACK solve: factor info invalid in %s\n",Caller);break;	
			case UMFPACK_ERROR_invalid_system: printf("UMFPACK solve: invalid system in %s\n",Caller);break;			
			default: printf("UMFPACK solve: unknown error\n");break;
		}
		exit(0);
	}	
	
	umfpack_toc(Time);
	if (time!=NULL) *time += Time[0];
	
	umfpack_di_free_symbolic(&Pattern_info);
	umfpack_di_free_numeric(&Factor_info);
	free(A_col);
	free(A_ind);
	free(A_val);
	return SUCCESS;
}

void convert_sparse_to_UMFPACK_long(sparse_matrix* A,int col_num,SuiteSparse_long** Col_start,SuiteSparse_long** Indices,double** Values,int* Size){
	int i,j,k,s,N;
	double aji;
	sparse_matrix* AT = get_transpose(A,col_num);
	//int n = A->size;
	*Col_start = (SuiteSparse_long*)malloc((col_num+1)*sizeof(SuiteSparse_long));
	N = 0;
	for (i=0;i<col_num;i++) N += AT->Len[i];
	*Indices = (SuiteSparse_long*)malloc(N*sizeof(SuiteSparse_long));
	*Values = (double*)malloc(N*sizeof(double));
	
	k = 0;
	for (i=0;i<col_num;i++){
		(*Col_start)[i] = (SuiteSparse_long)k;
		for (j=0;j<AT->Len[i];j++){
			aji = AT->Values[i][j];
			if (aji!=0){
				(*Indices)[k] = (SuiteSparse_long)AT->Indices[i][j];
				(*Values)[k] = aji;
				k++;
			}
		}
	}
	(*Col_start)[col_num] = (SuiteSparse_long)k;
	*Size = k;
	if (k<N){
		*Indices = (SuiteSparse_long*)realloc(*Indices,k*sizeof(SuiteSparse_long));
		*Values = (double*)realloc(*Values,k*sizeof(double));
	}
	free_sparse(AT);
}

int multifrontal_solver_long(sparse_matrix* A,double* X,double* b,char* Caller,double* time){
	const SuiteSparse_long system_type = UMFPACK_A; 														 // default: solves Ax=b
	
	static double total_umfpack_time = 0;
	
	double Time[2];
	if (A==NULL) return round(total_umfpack_time);
	
	double Control [UMFPACK_CONTROL];
	umfpack_dl_defaults (Control) ;
	//Control[UMFPACK_PIVOT_TOLERANCE] = 1e-8;
	//Control [UMFPACK_SYM_PIVOT_TOLERANCE] = 1e-6;

	
	static void* Pattern_info = NULL;
	static void* Factor_info = NULL;
	
	SuiteSparse_long* A_col = NULL;
	SuiteSparse_long* A_ind = NULL;
	double* A_val = NULL;	
	int el_number = 0;
	
	SuiteSparse_long status;
	int n = A->size;
	
	umfpack_tic(Time);
	
	convert_sparse_to_UMFPACK_long(A,n,&A_col,&A_ind,&A_val,&el_number);
	status = umfpack_dl_symbolic(n,n,A_col,A_ind,A_val,&Pattern_info,Control,UMFPACK_pattern_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_ERROR_invalid_matrix: printf("UMFPACK symbolic: invalid matrix in %s\n",Caller);break;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK symbolic: memory insufficient in %s\n",Caller);break;
			case UMFPACK_ERROR_internal_error: printf("UMFPACK symbolic: internal error in %s\n",Caller);break;
			default: printf("UMFPACK symbolic: unknown error\n");break;
		}
		exit(0);
	}
	
	status = umfpack_dl_numeric(A_col,A_ind,A_val,Pattern_info,&Factor_info,Control,UMFPACK_factor_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK numeric: singular matrix in %s\n",Caller);				
				umfpack_di_free_symbolic(&Pattern_info);					
				free(A_col);
				free(A_ind);
				free(A_val);
				print_sparse(A);
				print_to_matrix_market_format(A,"/Home/damage/radszuwe/local/FILTLAN/DATA/singalur_matrix.mtx");
				print_vector(X,n);
				break;
				//return FAILED;
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK numeric: memory insufficient in %s\n",Caller);break;
			case UMFPACK_ERROR_invalid_Symbolic_object: printf("UMFPACK numeric: pattern info invalid in %s\n",Caller);break;	
			case UMFPACK_ERROR_different_pattern: printf("UMFPACK numeric: pattern info has changed in %s\n",Caller);break;			
			default: printf("UMFPACK symbolic: numeric error\n");break;
		}
		
		int j;
		void* Addr_buffer[1024];
		char** Func_names = NULL;
		int f = backtrace(Addr_buffer,1024);
		Func_names = backtrace_symbols(Addr_buffer,f);
		printf("\nin:\n");
		if (Func_names!=NULL){
			for (j=0;j<f;j++) printf("%s\n",Func_names[j]);
		}
		
		exit(0);
	}
	if (UMFPACK_factor_info[UMFPACK_RCOND]>max_cond_num) max_cond_num = UMFPACK_factor_info[UMFPACK_RCOND];
	if (UMFPACK_factor_info[UMFPACK_RCOND]<min_cond_num) min_cond_num = UMFPACK_factor_info[UMFPACK_RCOND];
	
	status = umfpack_dl_solve(system_type,A_col,A_ind,A_val,X,b,Factor_info,Control,UMFPACK_solve_info);
	if (status!=UMFPACK_OK){
		switch(status){
			case UMFPACK_WARNING_singular_matrix: 
				printf("UMFPACK solve: singular matrix\n");
				umfpack_di_free_symbolic(&Pattern_info);
				umfpack_di_free_numeric(&Factor_info);
				free(A_col);
				free(A_ind);
				free(A_val);
				return FAILED;				
			case UMFPACK_ERROR_out_of_memory: printf("UMFPACK solve: memory insufficient in %s\n",Caller);break;
			case UMFPACK_ERROR_invalid_Numeric_object: printf("UMFPACK solve: factor info invalid in %s\n",Caller);break;	
			case UMFPACK_ERROR_invalid_system: printf("UMFPACK solve: invalid system in %s\n",Caller);break;			
			default: printf("UMFPACK solve: unknown error\n");break;
		}
		exit(0);
	}	
	
	umfpack_toc(Time);
	if (time!=NULL) *time += Time[0];
	
	umfpack_dl_free_symbolic(&Pattern_info);
	umfpack_dl_free_numeric(&Factor_info);
	free(A_col);
	free(A_ind);
	free(A_val);
	return SUCCESS;
}

int multifrontal_solver_RN(sparse_matrix* A,double* X,double* b){
	return multifrontal_solver(A,X,b,"Reflective Newton",&UMFPACK_time_damage);
}

int multifrontal_solver_AS(sparse_matrix* A,double* X,double* b){
	return multifrontal_solver(A,X,b,"Active Set",&UMFPACK_time_damage);
}

double normal_distribution(double mean,double var){
	const int N = 1000;
	int i;
	double res = 0;
	for (i=0;i<N;i++) res += (double)rand()/RAND_MAX;
	res -= (double)N/2;
	res *= sqrt((double)12*var/N);
	res += mean;
	return res;
}

double* random_initial(double mean,double var,int size){
	int i;
	srand(time(NULL));
	double* Res = zero_vector(size);
	for (i=0;i<size;i++) Res[i] = normal_distribution(mean,var);
	return Res;
}

double* compute_scalar_function(double* X,int print_info,double eps){	
	const int max_iter = 100;
	int m = get_var_number2D();
	int n = glob_mesh.size;
	set_var_number2D(1);
	sparse_matrix* ID = set_matrix_Aij_00_2D(0,0,&insert_AV);
	sparse_matrix* L = sparse_zero(n);
	sparse_matrix* U = sparse_zero(n);
	incomplete_LU_factorization(ID,L,U);
	PCG_default(ID,L,U);
	double* Res = zero_vector(n);
	// solve 
	double r0 = matrix_residuum(ID,Res,X);
	if (r0>0){
		if (r0>0) pcg(Res,X,n,eps,max_iter);
		double r = matrix_residuum(ID,Res,X);		
		if (print_info) printf("residuum function: %e\n",r/r0);
	}
	// clean
	free_sparse(ID);
	free_sparse(L);
	free_sparse(U);
	set_var_number2D(m);
	return Res;
}

double* interface_energy_pot(double* C){
	int i;
	int n = glob_mesh.size;
	double* Res = generate_vector(n,1.);
	//for (i=0;i<n;i++) if (fabs(C[i])<0.8) Res[i] *= 5e-3;
	return Res;
}

double* interface_energy_pot_dc(double* C){
	int n = glob_mesh.size;
	double* Res = generate_vector(n,0.);
	return Res;
}

double* interface_energy_grad(double* C){
	int i;
	int n = glob_mesh.size;
	double* Res = generate_vector(n,1.);
	//for (i=0;i<n;i++) if (fabs(C[i])<0.8) Res[i] *= 5e-3;
	return Res;
}

double* interface_energy_grad_dc(double* C){
	int n = glob_mesh.size;
	double* Res = generate_vector(n,0.);
	return Res;
}


double* free_energy_damage_interface_dc(double* H,double* C){
	int i;
	
	int n = glob_mesh.size;
	double* Xi = interface_energy_pot_dc(C);
	double* Fsqr = zero_vector(n);												
	for (i=0;i<n;i++) Fsqr[i] = H[i]-1.;
	double* Res = sparse3D_bilinear(T_Source,Fsqr,Fsqr);
	vector_pseudo_mult(Res,Xi,n);
	scalar_mult(xi,Res,n);
	
	double* Kappa = damage_interface_tensor(H,C,PARTIAL1);
	double* Grad = sparse3D_bilinear_tensor_2D_right(Strain_grad,H,H);
	double* contr = nodewise_tensor_contract_2D(Kappa,Grad);
	vector_add(Res,contr,1./2.,n);
	
	/*double* Kappa = interface_energy_grad_dc(C);
	double* Grad = sparse3D_bilinear(T_Flux,H,H);
	vector_pseudo_mult(Grad,Kappa,n);
	vector_add(Res,Grad,kappa/2.,n);*/
	
	free(Xi);
	free(Fsqr);
	free(Kappa);
	free(Grad);
	free(contr);
	return Res;
}

double damage_self_potential(double* X,double* C,int n){
	int i;
	double* Q;
	
	if (C==NULL) Q = Glob_Pointer_C; else Q = C;
	double* Xi = interface_energy_pot(Q);
	double* Fsqr = zero_vector(n);
	for (i=0;i<n;i++) Fsqr[i] = X[i]-1.;			   						// (z-1)^2: X[i]-1.
	double* Res = sparse3D_bilinear(T_Source,Fsqr,Fsqr);
	double res = xi*scalar(Xi,Res,n);
	
	//double old = xi*sparse_bilinear(Fsqr,ID_damage,Fsqr);
	//printf("diff func=%e\n",fabs((res-old)/old));
	
	free(Xi);
	free(Res);
	free(Fsqr);
	return res;
}

double* damage_self_gradient(double* X,double* C,int n){
	int i;
	double* Q;
		
	if (C==NULL) Q = Glob_Pointer_C; else Q = C;
	double* Xi = interface_energy_pot(Q);
	double* Fsqr = zero_vector(n);												
	for (i=0;i<n;i++) Fsqr[i] = X[i]-1.;									// (z-1)^2: X[i]-1.
	double* Res = sparse3D_bilinear(T_Source,Xi,Fsqr);	
	scalar_mult(2.*xi,Res,n);
	
	/*double* Old = sparse_mult(ID_damage,Fsqr);
	scalar_mult(2.*xi,Old,n);
	vector_add(Old,Res,-1.,n);
	printf("diff grad=%e\n",euklid_norm(Old,n)/euklid_norm(Res,n));
	free(Old);*/
	
	free(Xi);
	free(Fsqr);
	return Res;
}

sparse_matrix* damage_self_hesse(double* X,double* C,int n){	
	double* Q;
	
	if (C==NULL) Q = Glob_Pointer_C; else Q = C;	   	
	double* Xi = interface_energy_pot(Q);
	sparse_matrix* Res = sparse3D_Bv(T_Source,Xi);							// (z-1)^2
	scalar_sparse_mult(2.*xi,Res);
	
	/*sparse_matrix* Old = clone(ID_damage);
	scalar_sparse_mult(2.*xi,Old);
	sparse_add(Old,Res,-1.);
	printf("diff hess=%e\n",matrix_norm(Old)/matrix_norm(Res));
	free_sparse(Old);*/
	
	free(Xi);
	return Res;
}

double damage_C_interpolation(double c,double sqr_grad_c){
	//double fac;
	//if (fabs(c)<reduce_border_thougness_tol) fac = a_interface; else fac = 1.;
	double fac = exp(-a_interface*sqr_grad_c);
	if (fac<interface_thougness_min) fac = interface_thougness_min;
	return fac*((a_1+a_2)/2.+(a_1-a_2)*linear_limited(c)/2.);
}

double* damage_rate_independent(double* C){
	const double lx = 0.5;
	const double ly = 1.;
	
	int i;
	double x,y;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* Grad2 = sparse3D_Bvv(T_Flux,C,C);
	vector_pseudo_div(Grad2,V,n);
	
	double* Alpha = zero_vector(n);
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		
		#ifdef X_MARGIN
			if (x>-lx+margin_large && x<lx-margin_large) Alpha[i] = damage_C_interpolation(C[i],Grad2[i]);		// x-margin
			else Alpha[i] = large_toughness;			
		#endif
		
		#ifdef Y_MARGIN
			if (y>margin_large && y<ly-margin_large) Alpha[i] = damage_C_interpolation(C[i],Grad2[i]);						// y-margin
			else Alpha[i] = large_toughness;
		#endif 
		
		#if !defined X_MARGIN && !defined Y_MARGIN
			Alpha[i] = damage_C_interpolation(C[i],Grad2[i]);													// no margin
		#endif
	}
	
	free(V);
	free(Grad2);
	return Alpha;
}

double* get_sqr_grad(double* C){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* Grad2 = sparse3D_Bvv(T_Flux,C,C);
	vector_pseudo_div(Grad2,V,n);
	free(V);
	return Grad2;
}

double damage_functional(double* X,int n){
	if (Strain_energy!=NULL && Glob_Pointer_C!=NULL && Glob_Pointer_H0!=NULL){
		int i;
		double res;
		int time_index = glob_time_index;
		double* H_prev = Glob_Pointer_H0;
		
		// employ zero Dirichlet conditions
		double* H = clone_vector(X,n);
		for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET) H[i] = 0;
		
		// rate independent part
		double* Alpha = damage_rate_independent(Glob_Pointer_C);
		
		//double* Kappa = interface_energy_grad(Glob_Pointer_C);
		//double* S = sparse3D_bilinear(T_Flux,X,X);
		double* Kappa = damage_interface_tensor(Glob_Pointer_H0,Glob_Pointer_C,PARTIAL0);
		double* Grad = sparse3D_bilinear_tensor_2D_right(Strain_grad,X,X);
		double* S = nodewise_tensor_contract_2D(Kappa,Grad);
		res = dt*vector_contraction(S,n)/2.;
		//res = dt*kappa*scalar(Kappa,S,n)/2.;
		free(Grad);
		
		res -= dt*sparse_bilinear(H,Dissipation,Alpha);
		
		// elastic part
		double* G = damage_weight_function(H);
		res += dt*scalar(G,Strain_energy,n)/2.;
		
		//rate dependent part		
		double* V = clone_vector(X,n);
		if (time_index>0) vector_add(V,H_prev,-1.,n);
		else vector_add(V,V,-1.,n);
		res += beta*sparse_bilinear(V,ID_damage,V)/(2.);
		free(V);
				
		// self part
		res += dt*damage_self_potential(X,Glob_Pointer_C,n);
		
		free(Alpha);		
		free(Kappa);
		free(H);
		free(S);
		free(G);
		return res;
	}
	else{
		printf("error: no strain energy set -> abort\n");
		exit(0);
		return 0;
	}
}

double* damage_gradient(double* X){
	if (Strain_energy!=NULL && Glob_Pointer_C!=NULL && Glob_Pointer_H0!=NULL){
		int i;
		int n = glob_mesh.size;
		double* H_prev = Glob_Pointer_H0;
		
		// rate independent part
		double* Alpha = damage_rate_independent(Glob_Pointer_C);
		
		//double* Kappa = interface_energy_grad(Glob_Pointer_C);
		//double* Grad = sparse3D_vvB(T_Flux,Kappa,X,n);
		double* Kappa = damage_interface_tensor(Glob_Pointer_H0,Glob_Pointer_C,PARTIAL0);
		double* Grad = sparse3D_times_rank2_times_scalar_2D_right(Strain_grad,Kappa,X);
			
		scalar_mult(dt,Grad,n);
		linear_map(Grad,-dt,Dissipation,Alpha);		
		
		// elastic part
		double* G = damage_weight_function_dH(X);
		vector_pseudo_mult(G,Strain_energy,n);
		vector_add(Grad,G,dt/2.,n);
		free(G);
		
		// rate dependent part
		int time_index = glob_time_index;
		
		double* V = clone_vector(X,n);
		if (time_index>0) vector_add(V,H_prev,-1.,n);
		else vector_add(V,V,-1.,n);
		linear_map(Grad,beta,ID_damage,V);
		free(V);
		
		// self part
		double* W = damage_self_gradient(X,Glob_Pointer_C,n);							// (z-1)^2
		vector_add(Grad,W,dt,n);
		free(W);
		
		// employ zero Dirichlet conditions
		for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET) Grad[i] = X[i];		
	
		//scalar_mult(dt,G,n);
		free(Alpha);
		free(Kappa);
		return Grad;
	}
	else{
		printf("error: no strain energy set -> abort\n");
		exit(0);
		return NULL;
	}
}

double* neighbour_smoother(double* X){
	//int i,j,k;
	//double sum,d;
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	
	double* Res = sparse_mult(ID_damage,X);
	vector_pseudo_div(Res,V,n);
	free(V);
	
	/*double* Res = zero_vector(n);
	for (i=0;i<n;i++){
		sum = 0;
		for (i=0;i<ID_damage->Len[i];
		
		
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			
		}
	}*/
	
	return Res;
}



double* damage_driving_force(double* H,double* H_prev,double* C){
	if (Strain_energy!=NULL){
		int i;
		int n = glob_mesh.size;
		double* H_prev;
		
		// rate independent part
		double* Alpha = damage_rate_independent(C);
		
		//double* Kappa = interface_energy_grad(C);
		//double* Grad = sparse3D_vvB(T_Flux,Kappa,H,n);
		double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
		double* Grad = sparse3D_times_rank2_times_scalar_2D_right(Strain_grad,Kappa,H);
		//scalar_mult(kappa,Grad,n);
		free(Kappa);
		
		/*double* Old = sparse_mult(Interface_damage,H);		
		vector_add(Old,Grad,-1.,n);
		printf("diff:%e\n",euklid_norm(Old,n)/euklid_norm(Grad,n));*/
		
		linear_map(Grad,-1.,Dissipation,Alpha);		
		
		// elastic part
		double* G = damage_weight_function_dH(H);
		vector_pseudo_mult(G,Strain_energy,n);
		vector_add(Grad,G,(double)1/2,n);
		free(G);
		
		// rate dependent part
		int time_index = glob_time_index;
		
		/*double* V = clone_vector(H,n);
		if (time_index>0) vector_add(V,H_prev,-1.,n);
		else vector_add(V,V,-1.,n);
		linear_map(Grad,beta/dt,ID_damage,V);
		free(V);*/
		
		// self part
		double* W = damage_self_gradient(H,C,n);										// (z-1)^2
		vector_add(Grad,W,1.,n);
		free(W);
		
		// employ zero Dirichlet conditions
		//for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET) Grad[i] = H[i];		
	
		//scalar_mult(dt,G,n);
		free(Alpha);
		
		double* A = set_vector_Ui_0_2D(1);
		scalar_mult(-1./beta,Grad,n);
		vector_pseudo_div(Grad,A,n);
		free(A);
		
		//double* Res = neighbour_smoother(Grad);
		double* Res = clone_vector(Grad,n);
		
		free(Grad);
		return Res;
	}
	else{
		printf("damage driving force error: no strain energy set -> abort\n");
		exit(0);
		return NULL;
	}
}

sparse_matrix* get_damage_Hesse_linear(double* C){
	int n = glob_mesh.size;
	
	double* Sum = interface_energy_pot(C);												// self part xi*(z-1)^2
	scalar_mult(2.*xi*dt,Sum,n);											
	double* Rate_dep = generate_vector(n,beta);											// rate dependent part
	vector_add(Sum,Rate_dep,1.,n);	
	sparse_matrix* Res = sparse3D_Bv(T_Source,Sum);
	
	double* Kappa = damage_interface_tensor(Glob_Pointer_H0,Glob_Pointer_C,PARTIAL0);	// rate independent part
	sparse_matrix* Grad = sparse3D_times_rank2_2D_right(Strain_grad,Kappa);						
	
	sparse_add(Res,Grad,dt);
						
	free_sparse(Grad);
	free(Kappa);
	free(Sum);
	free(Rate_dep);
	return Res;
}

sparse_matrix* damage_Hesse(double* X){
	if (Strain_energy!=NULL && Damage_Hesse_linear!=NULL){
		int i;
		sparse_matrix* Res;
		
		int n = glob_mesh.size;
		
		double* G = damage_weight_function_ddH(X);
		vector_pseudo_mult(G,Strain_energy,n);
		scalar_mult(dt,G,n);
		
		if (euklid_norm(G,n)!=0) Res = sparse3D_Bv(T_Source,G); else Res = sparse_zero(n);
		sparse_add(Res,Damage_Hesse_linear,1.);			
	
		for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET){
			reset_row(Res,i);
			insert_sparse(Res,1.,i,i);
		}
			
		free(G);
		return Res;
	}
	else{
		printf("error: no strain energy set -> abort\n");
		exit(0);
		return NULL;
	}
}

double func(double* X,int m){
	int i;
	set_var_number2D(1);
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV);
	sparse_matrix* B = set_matrix_Aij_00_2D(0,0,&insert_AV);	
	double* b = generate_vector(m,-1.);
	for (i=0;i<m;i++) if (R_Newton_Condition[i]==DIRICHLET){
		reset_row(A,i);
		reset_row(B,i);
		insert_sparse(A,1.,i,i);
		insert_sparse(B,0,i,i);
	}
	
	double res = sparse_bilinear(X,A,X)/2.+sparse_bilinear(b,B,X);
	
	free_sparse(A);
	free_sparse(B);
	free(b);
	return res;
}

double* grad(double* X){
	int i;
	int n = glob_mesh.size;
	set_var_number2D(1);
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV);
	sparse_matrix* B = set_matrix_Aij_00_2D(0,0,&insert_AV);
	
	double* b = generate_vector(n,-1.);
	for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET){
		reset_row(A,i);
		reset_row(B,i);
		insert_sparse(A,1.,i,i);
		insert_sparse(B,0,i,i);
	}
	
	double* Res = sparse_mult(B,b);
	linear_map(Res,1.,A,X);
	
	free_sparse(A);
	free_sparse(B);
	free(b);
	return Res;
}

sparse_matrix* hess(double* X){
	int i;
	int n = glob_mesh.size;
	set_var_number2D(1);
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV);
	
	for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET){
		reset_row(A,i);	
		insert_sparse(A,1.,i,i);
	}
	
	return A;
}

int RN_vs_AS(double* X){
	int i;
	int n = glob_mesh.size;
	double* Y = clone_vector(X,n);
	
	// constraint solver settings
	R_Newton_Delta = sqrt((double)n)/2.;
	R_Newton_trust_radius_max = 2.*R_Newton_Delta;
	RN_set_params(R_Newton_max_iter,R_Newton_eps,R_Newton_sigma_l,R_Newton_sigma_u,R_Newton_rho,R_Newton_Delta,
	 R_Newton_Xi,R_Newton_trust_lower,R_Newton_trust_upper,R_Newton_trust_radius_min,R_Newton_trust_radius_max);
	
	RN_set_local_solver(&multifrontal_solver_RN);							 // set Multifrontal solver
	R_Newton_Condition = zero_int_list(n);
	for (i=0;i<n;i++) if (is_boundary(i)>=0) R_Newton_Condition[i] = DIRICHLET;
	RN_set_Hesse_function(&hess,n);
	RN_set_gradient(&grad);
	set_RN_functional(&func);

	RN_set_trust_region(R_Newton_Delta);	
	int* Indices = get_all_indices(n);
	double* Lower = generate_vector(n,-5e-2);
	double* Upper = generate_vector(n,5e-2);
	set_RN_constraints(Indices,Lower,Upper,n);									
	//RN_set_kernel(Kernel_step2,n);
	int iter1 = reflective_newton(X);	
	
	print_scalar_data("/Home/damage/radszuwe/Daten/LSGNewton",X,n);
	printf("reflective Newton: %d iterations\n",iter1);
	

	set_var_number2D(1);
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV);
	sparse_matrix* B = set_matrix_Aij_00_2D(0,0,&insert_AV);
	double* CD = generate_vector(n,1.);
	sparse_matrix* C = sparse_diagonal(CD,n);
	
	for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET){
		reset_row(A,i);
		reset_row(B,i);
		insert_sparse(A,1.,i,i);
		insert_sparse(B,0,i,i);
	}
	double* b = generate_vector(n,-1.);
	sparse_multiplication(B,b);
	scalar_mult(-1.,b,n);
	
	ass_info* info = NULL;
	int iter2 = active_set_solver(A,Y,b,C,Lower,Upper,&multifrontal_solver_RN,R_Newton_max_iter
	 ,POS_DEFINITE,ASS_DEFAULT,&info);
	free_ass_info(&info);
	
	print_scalar_data("/Home/damage/radszuwe/Daten/LSGActive",Y,n);
	printf("PD Active Set: %d iterations\n",iter2);
	
	printf("solution diff: %e\n",vec_dist(X,Y,n)/euklid_norm(X,n));
	
	return 0;
}

	const double ga = -5.2;
	const double gb = -7.27;
	const double gc = 9.23e-4;
	const double xi_1 = 2.97;
	const double xi_2 = 3.01;
	
double eutectic_map(double c){
	return (c-c_eutectic)/(1.-c_eutectic*c);
}

double eutectic_map_dc(double c){
	double d = 1.-c_eutectic*c;
	return (1.-c_eutectic*c_eutectic)/(d*d);
}

double eutectic_map_ddc(double c){
	double d = 1.-c_eutectic*c;
	return -2.*c_eutectic*(1.-c_eutectic*c_eutectic)/(d*d*d);
}

double Landau_potential(double y,double T){
	double y2 = y;
	y2 *= y;
	return pot_factor*y2*(y2/2.+(T+T_absolute_ref-T_crit)/T_crit)/2.;
}

double Landau_potential_dy(double y,double T){
	double y2 = y;
	y2 *= y;
	return pot_factor*y*(y2+(T+T_absolute_ref-T_crit)/T_crit);
}

double Landau_potential_dT(double y,double T){
	return pot_factor*y*y/(2.*T_crit);
}

double Landau_potential_ddy(double y,double T){
	return pot_factor*(3.*y*y+(T_absolute_ref-T_crit)/T_crit);
}

double Landau_potential_dydT(double y,double T){
	return pot_factor*y/T_crit;
}


sparse_matrix* get_CH_nonlinear_part(double* H,double* Y,double* T){		//	\partial_c((c^2-1)^2)+\na*(m\na\mu)
	int i;
	double c,y,dy,ddy;
	
	int n = glob_mesh.size;
	double* C = &(Y[0]);
	
	// Landau potential
	double* F = zero_vector(2*n);
	if (T!=NULL){		
		for (i=0;i<n;i++){
			y = eutectic_map(C[i]);
			dy = eutectic_map_dc(C[i]);
			ddy = eutectic_map_ddc(C[i]);
			F[i] = Landau_potential_ddy(y,T[i])*dy*dy+Landau_potential_dy(y,T[i])*ddy;
			//F[i] = pot_factor*(3.*C[i]*C[i]+((T[i]+T_absolute_ref-T_crit)/T_crit));
			
		}
		/*for (i=0;i<n;i++){																	// Attention: When this potential is activated 
			c = (1.+C[i])/2.;																	// the C values have to be limited in alternate minimization 
																								// and after interpolation ! 	  
			F[i] = pot_factor*(gc*(T_absolute_ref+T[i])/(c*(1.-c))-2.*(xi_1*c+xi_2*(1.-c))+2.*(1.-2.*c)*(xi_1-xi_2))/4.;
		}*/
	}
	else{
		F[i] = 4.*(3.*C[i]*C[i]-1.);
	}
	sparse_matrix* Res = sparse3D_Bv(Phase_Jacobian,F);
	
	free(F);
	return Res;
}

double* Jacobian_self_c(double* C,double* T){		
	int i;
	double c,y;
	int n = glob_mesh.size;
	double* Res = zero_vector(2*n);
	if (T!=NULL){
		for (i=0;i<n;i++){
			y = eutectic_map(C[i]);
			Res[i] = Landau_potential_dy(y,T[i])*eutectic_map_dc(C[i]);
			//Res[i] = pot_factor*(C[i]*(C[i]*C[i]+(T[i]+T_absolute_ref-T_crit)/T_crit));			
		}
		/*for (i=0;i<n;i++){
			c = (1.+C[i])/2.;	
			Res[i] = 	pot_factor*(gb-ga+gc*(T_absolute_ref+T[i])*log(c/(1.-c))+(1.-2.*c)*(xi_1*c+xi_2*(1.-c))+c*(1.-c)*(xi_1-xi_2))/2.;
		}*/
	}
	else{
		Res[i] = 4.*C[i]*(C[i]*C[i]-1.);
	}
	sparse_multiplication(Phase_ID_c,Res);
	for (i=0;i<n;i++){
		Res[i+n] = Res[i];
		Res[i] = 0;
	}
	return Res;
}

double* free_energy_self_c(double* C,double* T){									// one-dimensional, not multiplied by \phi_i
	int i;
	double c,y;
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	if (T!=NULL){
		for (i=0;i<n;i++){
			y = eutectic_map(C[i]);			
			Res[i] = Landau_potential(y,T[i]);
			//pot_factor*(C[i]*C[i]*(C[i]*C[i]/4.+(T[i]+T_absolute_ref-T_crit)/(2.*T_crit)));
		}
		/*for (i=0;i<n;i++){
			c = (1.+C[i])/2.;			
			Res[i] = pot_factor*(ga*(1.-c)+gb*c+gc*(T_absolute_ref+T[i])*(c*log(c)+(1.-c)*log(1.-c))+c*(1.-c)*(xi_1*c+xi_2*(1.-c)));
		}*/
	}
	else{
		Res[i] = (C[i]*C[i]-1.)*(C[i]*C[i]-1.);
	}
	return Res;
}

double* free_energy_self_c_dT(double* C,double* T){
	int i;
	double c,y;
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	if (T!=NULL){
		for (i=0;i<n;i++){
			y = eutectic_map(C[i]);
			Res[i] = Landau_potential_dT(y,T[i]);
			//Res[i] = pot_factor*C[i]*C[i]/(2.*T_crit);			
		}
		/*for (i=0;i<n;i++){
			c = (1.+C[i])/2.;	
			Res[i] = pot_factor*gc*(c*log(c)+(1.-c)*log(1.-c));
		}*/
	}
	return Res;
}

double* free_energy_self_c_dT_dC(double* C,double* T){
	int i;
	double c,y,dy;
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	if (T!=NULL){
		for (i=0;i<n;i++){
			y = eutectic_map(C[i]);
			dy = eutectic_map_dc(C[i]);
			Res[i] = Landau_potential_dydT(y,T[i])*dy;
			//Res[i] = pot_factor*C[i]/T_crit;
		}
		/*for (i=0;i<n;i++){
			c = (1.+C[i])/2.;	
			Res[i] = pot_factor*gc*log(c/(1.-c))/2.;
		}*/
	}
	return Res;
}

void reflect(double* C,double lower,double upper){
	int i,index;
	double w,dummy;
	int n = glob_mesh.size;
	double d = 2.*(upper-lower);
	for (i=0;i<n;i++){			
		w = modf(fabs(C[i]-lower)/d,&dummy)*d;
		C[i] = (w<d-w) ? lower+w : lower+d-w;
	}
}

void set_dirichlet_BC(sparse_matrix* A,double* F,int* List,double (*Bound_func)(int node,int var_index, double t)){
	int i,j;
	int d = glob_mesh.size;
	int deg = A->size / d;
	for (i=0;i<d;i++) if (List[i]==DIRICHLET){
		for (j=0;j<deg;j++){
			if (F!=NULL) F[i+j*d] = (*Bound_func)(i,j,glob_time);
			reset_row(A,i+j*d);
			insert_sparse(A,1.,i+j*d,i+j*d);
		}
	}
}

void Set_dirichlet_BC(sparse_matrix* A,double* F,bound_cond* BC){
	int i;
	
	int n = A->size;
	if (n!=BC->size){
		printf("Set_dirichlet_BC: sizes do not match -> abort\n");
		exit(0);
	}
	
	for (i=0;i<n;i++) if (BC->Cond[i]==DIRICHLET){
		if (F!=NULL) F[i] = BC->Val[i];
		reset_row(A,i);
		insert_sparse(A,1.,i,i);
	}
}

double* get_bound_force(bound_cond* BC){
	
	int i;
	int n = glob_mesh.size;
	double* Res = zero_vector(dimension*n);
	for (i=0;i<dimension*n;i++) if (BC->Cond[i]==NEUMANN || BC->Cond[i]==ROBIN) Res[i] = BC->Val[i]; 
	return Res;
}

void Set_Robin_BC(sparse_matrix* A,double* F,bound_cond* BC){
	int i,j,k,l;
	int n = A->size;
	if (n!=BC->size){
		printf("Set_ROBIN_BC: sizes do not match -> abort\n");
		exit(0);
	}
	
	sparse_matrix* B = clone(Bound_Neumann);
	for (i=0;i<n;i++){
		if (BC->Cond[i]!=ROBIN) reset_row(B,i);
		else{
			
		}
	}
	
	int N = glob_mesh.size;
	double* D = zero_vector(n);
	for (i=0;i<N;i++){
		D[i] = Robin_stiffness.x;
		D[N+i] = Robin_stiffness.y;
	}
	
	for (i=0;i<n;i++){
		for (j=0;j<B->Len[i];j++) B->Values[i][j] *= D[i];
	}
	
	sparse_add(A,B,1.);
	
	if (F!=NULL){
		double* b = zero_vector(n);
		for (i=0;i<n;i++) b[i] = BC->Val[i];
		linear_map(F,1.,B,b);
		free(b);
	}
	free_sparse(B);
}

double* Two_phase_interpolation(double factor1,double factor2,double* C,int n){			// returns factor1*(1+c)/2+factor2*(1-c)/2
	double* Res = generate_vector(n,(factor1+factor2)/2.);							
	vector_add(Res,C,(factor1-factor2)/2.,n);
	return Res;
}

double* Metric(double* U){
	
	double* V = set_vector_Ui_0_2D(1);
	
	set_var_number2D(dimension);
	sparse_matrix* Aabx = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_x);
	sparse_matrix* Aaby = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_y);
	sparse_matrix* Abax = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_x);
	sparse_matrix* Abay = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_y);
	sparse_matrix3D* Bx = set_matrix_Bijk_011(0,0,0,&insert_B_abV_cV_c_rowx);
	sparse_matrix3D* By = set_matrix_Bijk_011(0,0,0,&insert_B_abV_cV_c_rowy);

	int i,j,k;
	double* U2x;
	double* U2y;
	int n = glob_mesh.size;
	double* Res = zero_vector(dimension*dimension*n);
	for (i=0;i<dimension;i++){															// delta_ab
		for (j=0;j<dimension;j++) if (i==j){
			for (k=0;k<n;k++) Res[(i*dimension+j)*n+k] = V[k];
		} 
	}
	
	linear_map(&(Res[0]),1.,Aabx,U);													// u_a|b+u_b|a
	linear_map(&(Res[0]),1.,Abax,U);
	linear_map(&(Res[dimension*n]),1.,Aaby,U);
	linear_map(&(Res[dimension*n]),1.,Abay,U);
	
	
	U2x = sparse3D_bilinear(Bx,U,U);													// u_c|a*u_c|b
	vector_add(&(Res[0]),U2x,1.,2*n);
	U2y = sparse3D_bilinear(By,U,U);
	vector_add(&(Res[dimension*n]),U2y,1.,2*n);
	
	for (i=0;i<dimension;i++){
		for (j=0;j<dimension;j++) vector_pseudo_div(&(Res[(i*dimension+j)*n]),V,n);			
	}
	
	free_sparse3D(Bx);
	free_sparse3D(By);
	free_sparse(Aabx);
	free_sparse(Aaby);
	free_sparse(Abax);
	free_sparse(Abay);
	free(V);
	free(U2x);
	free(U2y);
	
	return Res;
}

double* Deformation_gradient(double* U){
	int i,j,k;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	set_var_number2D(dimension);
	sparse_matrix* Aabx = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_x);
	sparse_matrix* Aaby = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_y);
	
	double* Res = zero_vector(dimension*dimension*n);
	for (i=0;i<dimension;i++){															// delta_ab
		for (j=0;j<dimension;j++) if (i==j){
			for (k=0;k<n;k++) Res[(i*dimension+j)*n+k] = V[k];
		} 
	}
	
	linear_map(&(Res[0]),1.,Aabx,U);													// u_a|b
	linear_map(&(Res[dimension*n]),1.,Aaby,U);

	for (i=0;i<dimension;i++){
		for (j=0;j<dimension;j++) vector_pseudo_div(&(Res[(i*dimension+j)*n]),V,n);			
	}
	
	free_sparse(Aabx);
	free_sparse(Aaby);
	free(V);
	return Res;
}

double* Strain_tensor(double* U){
	int i,j,k;
	int n = glob_mesh.size;
	
	double* E = Metric(U);															
	for (i=0;i<dimension;i++){											
		for (j=0;j<dimension;j++) if (i==j){
			for (k=0;k<n;k++) E[(i*dimension+j)*n+k] -= 1.;
		} 
	}
	scalar_mult((double)1/2,E,dimension*dimension*n);							
	return E;
}

double* first_invariant(double* G){		// G[0] -> xx
	int i;								// 1 -> xy
	int n = glob_mesh.size;				// 2 -> yx
	double* Res = zero_vector(n);		// 3 ->yy
	for (i=0;i<n;i++) Res[i] = G[i]+G[i+3*n];
	return Res;
}

double* third_invariant(double* G){
	int i;					
	int n = glob_mesh.size;
	double* Res = zero_vector(n);
	for (i=0;i<n;i++) Res[i] = G[i]*G[i+3*n]-G[i+n]*G[i+2*n];
	return Res;
}

double* second_invariant(double* G){											// in 2D: II=III
	return third_invariant(G);
}

double* get_elastic_stress(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(4);
	
	double* rank4_modulus = elastic_constitutive(U,H,C,T,PARTIAL0,PARTIAL0);
	double* E = linear_strain_tensor(U);
	double* S = linear_rank2(rank4_modulus,E);
	vector_pseudo_div(S,V,4*n);
	
	free(rank4_modulus);
	free(E);
	free(V);
	return S;
}

double* get_total_stress(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(4);
	
	double* rank4_modulus = elastic_constitutive(U,H,C,T,PARTIAL0,PARTIAL0);
	double* E = linear_strain_tensor(U);
	double* Strain = total_strain(E,C,T,PARTIAL0,PARTIAL0);
	double* Stress = linear_rank2(rank4_modulus,Strain);
	vector_pseudo_div(Stress,V,4*n);
	
	free(rank4_modulus);
	free(E);
	free(Strain);
	free(V);
	return Stress;
}

double Free_energy_heat(double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double ra = (rho_1+rho_2)/2.;
	double rb = (rho_1-rho_2)/2.;
	
	double* X = clone_vector(T,n);
	vector_shift(X,n,T_absolute_ref);
	double* Y = zero_vector(n);
	for (i=0;i<n;i++) Y[i] = log(X[i])*(ra+rb*linear_limited(C[i]));
	double res = -sparse_bilinear(Y,ID_damage,X);
	
	free(X);
	free(Y);
	return res;
}

double* free_energy_heat(double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double ra = (rho_1+rho_2)/2.;
	double rb = (rho_1-rho_2)/2.;
	
	double* X = clone_vector(T,n);
	vector_shift(X,n,T_absolute_ref);
	double* Y = zero_vector(n);
	for (i=0;i<n;i++) Y[i] = -log(X[i])*(ra+rb*linear_limited(C[i]));
	double* Res = sparse_mult(ID_damage,Y);
	
	free(X);
	free(Y);
	return Res;
}

double* free_energy_heat_dc(double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double rb = (rho_1-rho_2)/2.;
	
	double* X = clone_vector(T,n);
	vector_shift(X,n,T_absolute_ref);
	double* Y = zero_vector(n);
	for (i=0;i<n;i++) Y[i] = -rb*linear_limited_derivative(C[i])*log(X[i])*X[i];
	double* Res = sparse_mult(ID_damage,Y);
	
	free(X);
	free(Y);
	return Res;
}

double* free_energy_heat_dT(double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double ra = (rho_1+rho_2)/2.;
	double rb = (rho_1-rho_2)/2.;
	
	double* X = clone_vector(T,n);
	vector_shift(X,n,T_absolute_ref);
	double* Y = zero_vector(n);
	for (i=0;i<n;i++) Y[i] = -(ra+rb*linear_limited(C[i]))*(1.+log(X[i]));
	double* Res = sparse_mult(ID_damage,Y);
	
	free(X);
	free(Y);
	return Res;
}

double* free_energy_heat_dcdT(double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double rb = (rho_1-rho_2)/2.;
	
	double* X = clone_vector(T,n);
	vector_shift(X,n,T_absolute_ref);
	double* Y = zero_vector(n);
	for (i=0;i<n;i++) Y[i] = -rb*linear_limited_derivative(C[i])*(1.+log(X[i]));
	double* Res = sparse_mult(ID_damage,Y);
	
	free(X);
	free(Y);
	return Res;
}


double* linear_strain_tensor(double* U){
	int i;
	int n = glob_mesh.size;
	
	double* JU = zero_vector(4*n);
	copy_vector_content(U,JU,0,0,2*n);
	copy_vector_content(U,JU,0,2*n,2*n);
	sparse_multiplication(Strain_tensor_Ax,&(JU[0]));
	sparse_multiplication(Strain_tensor_Ay,&(JU[2*n]));
	
	double* E = clone_vector(JU,4*n);
	for (i=0;i<n;i++){
		E[n+i] = (JU[n+i]+JU[2*n+i])/2.;
		E[2*n+i] = E[n+i];
	}
	
	free(JU);
	return E;
}

double* total_strain(double* Strain,double* C,double* T,int order_dc,int order_dT){
	int i;
	double* Res;
	
	int n = glob_mesh.size;
	if (Strain!=NULL) Res = clone_vector(Strain,dimension*dimension*n); else Res = zero_vector(dimension*dimension*n);
	double* S = self_strain(C,T,order_dc,order_dT);
	for (i=0;i<dimension*dimension;i++) linear_map(&(Res[i*n]),-1.,ID_damage,&(S[i*n]));
	free(S);
	return Res;
}

double* chemical_potential(double* U,double* H,double* C,double* T){			// mu = dF/dc
	int i;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* Res = zero_vector(n);
	
	// elastic part
	double* E = linear_strain_tensor(U);
	double* S = total_strain(E,C,T,PARTIAL0,PARTIAL0);
	double* dS = total_strain(NULL,C,T,PARTIAL1,PARTIAL0);
	
	double* Stress1 = constitutive_stress(dS,U,H,C,T,PARTIAL0,PARTIAL0);
	double* Stress2 = constitutive_stress(S,U,H,C,T,PARTIAL0,PARTIAL1);	
	vector_add(Stress1,Stress2,(double)1/2,n*dimension*dimension);
	for (i=0;i<n;i++) Res[i] = S[i]*Stress1[i]+S[n+i]*Stress1[n+i]+S[2*n+i]*Stress1[2*n+i]+S[3*n+i]*Stress1[3*n+i];
	vector_pseudo_div(Res,V,n);	
	
	free(E);
	free(S);
	free(dS);
	free(Stress1);
	free(Stress2);
	
	//damage part 
	double* D = free_energy_damage_interface_dc(H,C);
	vector_add(Res,D,1.,n);
	free(D);
	
	// phase part
	double* Phi_c = Jacobian_self_c(C,T);
	vector_add(Res,&(Phi_c[n]),1.,n);
	free(Phi_c);
	
	// interface part
	linear_map(Res,-kappa_c,Interface_chemical,C);
	
	// heat part
	double* Phi_T = free_energy_heat_dc(C,T);
	vector_add(Res,Phi_T,1.,n);
	free(Phi_T);
	
	free(V);
	return Res;
}

double* entropy(double* U,double* H,double* C,double* T){						// S=-dF/dT
	int n = glob_mesh.size;
	
	// elastic part
	double* Res = elastic_potential_self(U,H,C,T,PARTIAL0,PARTIAL0,PARTIAL1);
	scalar_mult(-1.,Res,n);
	
	// phase part
	double* SC = free_energy_self_c_dT(C,T);
	linear_map(Res,-1.,ID_damage,SC);
	free(SC);
	
	// heat part
	double* ST = free_energy_heat_dT(C,T);
	vector_add(Res,ST,-1.,n);
	free(ST);
	
	return Res;
}

double Total_entropy(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	double* E = generate_vector(n,1.);
	double* S = entropy(U,H,C,T);
	double res = scalar(E,S,n);
	free(E);
	free(S);
	return res;
}

/*double* damage_potential(double* U,double* H,double* C,double* T){						// gamma = dF/dz
	int n = glob_mesh.size;
	
	// elastic part
	double* Res = elastic_potential(U,H,C,T,PARTIAL1,PARTIAL0);
	double* Self = elastic_self_potential(U,H,C,T,PARTIAL1,PARTIAL0,PARTIAL0);
	vector_add(Res,Self,1.,n);
	free(Self);
	
	// damage part
	double* G = damage_self_gradient(H,n);
	vector_add(Res,G,1.,n);
	free(G);
	
	// damage interface part
	linear_map(Res,-1.,Interface_damage,H);
	
	return Res;
}*/

double* sigma_dT(double* U,double* H,double* C,double* T){			// dsigma/dT = d^2F/(dTde), returns vector of size 4*n
	int i;
	
	int n = glob_mesh.size;
	double* S_dT = self_strain(C,T,PARTIAL0,PARTIAL1);
	
	for (i=0;i<4;i++) sparse_multiplication(ID_damage,&(S_dT[n*i]));
	scalar_mult(-1.,S_dT,dimension*dimension*n);
	
	double* Rank4 = elastic_constitutive(U,H,C,T,PARTIAL0,PARTIAL0);
	double* Res = linear_rank2(Rank4,S_dT);
	
	free(S_dT);
	free(Rank4);
	return Res;
}

double* chemical_potential_dT(double* U,double* H,double* C,double* T){			// dmu/dT = d^2F/(dTdc)
	int i;
	int n = glob_mesh.size;

	// elastic part	
	double* Res = elastic_potential_self(U,H,C,T,PARTIAL0,PARTIAL1,PARTIAL1);
	
	// phase part
	double* F_dCdT = free_energy_self_c_dT_dC(C,T);
	linear_map(Res,1.,ID_damage,F_dCdT);
	free(F_dCdT);
	
	// heat part
	double* Phi_T = free_energy_heat_dcdT(C,T);
	vector_add(Res,Phi_T,1.,n);
	free(Phi_T);
	
	return Res;
}

double* damage_potential_dT(double* U,double* H,double* C,double* T){						// dgamma/dT = d^2F/(dTdz)

	// elastic part
	double* Res = elastic_potential_self(U,H,C,T,PARTIAL1,PARTIAL0,PARTIAL1);
	
	return Res;
}

sparse_matrix* heat_capacity(double* U,double* H,double* C,double* T){			// c = -T*d^2F/dT^2

	int i;
	int n = glob_mesh.size;
	double ra = (rho_1+rho_2)/2.;
	double rb = (rho_1-rho_2)/2.;
	double* V = set_vector_Ui_0_2D(1);
	
	// elastic part
	double* ddF = elastic_potential_self(U,H,C,T,PARTIAL0,PARTIAL0,PARTIAL2);
	for (i=0;i<n;i++) ddF[i] *= -(T[i]+T_absolute_ref)/V[i];
	
	// heat part
	for (i=0;i<n;i++){
		ddF[i] += ra+rb*linear_limited(C[i]);
		if (ddF[i]<0){
			printf("\nApproximation failed: heat capacity less than zero (c_v=%e, T=%e, C=%e H=%e)-> abort\n",ddF[i],T[i],C[i],H[i]);
			exit(0);
		} 
	}
	
	sparse_matrix* Res = sparse3D_Bv(T_Source,ddF);
	
	free(ddF);
	free(V);
	return Res;
}

double* heat_capacity_field(double* U,double* H,double* C,double* T){
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* E = generate_vector(n,1.);
	sparse_matrix* A = heat_capacity(U,H,C,T);
	double* Res = sparse_mult(A,E);
	vector_pseudo_div(Res,V,n);
	
	free_sparse(A);
	free(V);
	free(E);
	return Res;
}

double Boundary_entropy_flux(double* H,double* C,double* T){
	int i;
	double res,t;
	double t0 = T_outer+T_absolute_ref;
	
	int n = glob_mesh.size;
	double ka = (kappa_T1+kappa_T2)/2.-kappa_T0;
	double kb = (kappa_T1-kappa_T2)/2.;
	
	set_var_number2D(1);
	bound_cond* BC = get_T_bound_conditions(T,t);
	//int* Cond = get_T_condition_list();
	
	sparse_matrix* Grad = set_matrix_bound_2_Aij_10_2D(n_T,n_T,insert_A_bbV);
	
	double* F = zero_vector(n);
	for (i=0;i<n;i++) if (BC->Cond[i]==DIRICHLET) F[i] = 0; else F[i] = 1.;
	
	double* LogT = zero_vector(n);
	for (i=0;i<n;i++) LogT[i] = log(T_absolute_ref+T[i]);
	
	double* RecT = zero_vector(n);
	for (i=0;i<n;i++) RecT[i] = 1./(T_absolute_ref+T[i]);
	
	double* D = zero_vector(n);
	for (i=0;i<n;i++) D[i] = (kappa_T0+(ka+kb*linear_limited(C[i]))*H[i])*(1.-F[i]);
	res = sparse_bilinear(LogT,Grad,D);
	
	double* N = zero_vector(n);
	for (i=0;i<n;i++) if (BC->Cond[i]==NEUMANN || BC->Cond[i]==ROBIN){
		//t = T[i]+T_absolute_ref;
		//N[i] = (kappa_bound*(T_outer-T[i])+surface_emissivity*Boltz*(t0*t0*t0*t0-t*t*t*t))*F[i];
		N[i] = BC->Val[i];
	}
	res += sparse_bilinear(RecT,T_Neumann,N);
	
	free_bound_cond(&BC);
	free_sparse(Grad);
	//free(Cond);
	free(LogT);
	free(RecT);
	free(F);
	free(D);
	free(N);
	return res;
}

double* entropy_change_non_thermal_part(double* U,double* U_prev,double* H,		// computes -(dS/dt-c/T*dT/dt)*dt
	double* H_prev,double* C,double* C_prev,double* T){		
	
	int i;
	double* dF;
	
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* Res = zero_vector(n);
	
	// d^2F/(dedT)
	double* dE = linear_strain_tensor(U);
	double* E_prev = linear_strain_tensor(U_prev);
	vector_add(dE,E_prev,-1.,dimension*dimension*n);
	vector_pseudo_div(dE,V,n);
	dF = sigma_dT(U,H,C,T);
	for (i=0;i<n;i++) Res[i] += dF[i]*dE[i]+dF[n+i]*dE[n+i]+dF[2*n+i]*dE[2*n+i]+dF[3*n+i]*dE[3*n+i];
	free(dF);
	
	// d^2F/(dcdT)
	double* dC = clone_vector(C,n);
	vector_add(dC,C_prev,-1.,n);
	dF = chemical_potential_dT(U,H,C,T);
	vector_pseudo_mult(dF,dC,n);
	vector_add(Res,dF,1.,n);
	free(dF);
	
	// d^2F/(dzdT)
	double* dH = clone_vector(H,n);
	vector_add(dH,H_prev,-1.,n);
	dF = damage_potential_dT(U,H,C,T);
	vector_pseudo_mult(dF,dH,n);
	vector_add(Res,dF,1.,n);
	free(dF);
	
	free(V);
	free(E_prev);
	free(dE);
	free(dC);
	free(dH);
	return Res;
}

double Dissipation_chemical(double* Y,double* H,double* C_prev,double* C,double* T){
	int n = glob_mesh.size;
	
	double* M = get_mobility(H,C,T);
	double* YN = zero_vector(2*n);
	double* YT = zero_vector(2*n);
	copy_vector_content(Y,YN,0,n,n);
	copy_vector_content(Y,YT,0,0,n);
	//double res = mobility*sparse_bilinear(Y,Interface_chemical,Y)*dt;
	sparse_matrix* Grad2 = sparse3D_vB(Phase_Diffusion_mu,M,2*n);
	double res = sparse_bilinear(YT,Grad2,YN)*dt;
	free_sparse(Grad2);
	free(YN);
	free(YT);
	free(M);
	
	double* dC = clone_vector(C,n);
	vector_add(dC,C_prev,-1.,n);
	res += vis_c*sparse_bilinear(dC,ID_damage,dC)/dt;
	free(dC);
	return res;
}

double Entropy_production_rate(double* U,double* U_prev,double* H,double* H_prev,double* C,double* C_prev,
 double* Y,double* T,double* T_prev){	
	
	int i;
	double* S;
	int n = glob_mesh.size;
	double res = 0;
	double ka = (kappa_T1+kappa_T2)/2.-kappa_T0;
	double kb = (kappa_T1-kappa_T2)/2.;
	double* TRec = zero_vector(n);
	for (i=0;i<n;i++) TRec[i] = 1./(T[i]+T_absolute_ref);
	
	// thermal part
	double* LogT = zero_vector(n);
	for (i=0;i<n;i++) LogT[i] = log(T[i]+T_absolute_ref);
	double* K = zero_vector(n);
	for (i=0;i<n;i++) K[i] = kappa_T0+(ka+kb*C[i])*H[i];
	S = sparse3D_bilinear(T_Flux,LogT,LogT);
	
	res += scalar(K,S,n);							// bulk part
	res += Boundary_entropy_flux(H,C,T);			// boundary part
	free(S);
	
	// other part
	S = Heat_source(U,U_prev,H,H_prev,C,C_prev,Y,T);
	res += scalar(TRec,S,n)/dt;
	free(S);
	
	free(TRec);
	free(LogT);
	free(K);
	return res;
}

double Inner_Energy(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	
	double* Th = clone_vector(T,n);
	vector_shift(Th,n,T_absolute_ref);
	double* S = entropy(U,H,C,T);
	double res = Total_free_energy(U,H,C,T)+scalar(Th,S,n);
	
	free(Th);
	free(S);
	return res;
}

double Thermal_energy(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	
	sparse_matrix* CV = heat_capacity(U,H,C,T);
	double* UT = sparse_mult(CV,T);
	double res = vector_contraction(UT,n);
	
	free_sparse(CV);
	free(UT);
	return res;
}


double extremal_crack_position_y(double* H){
	const double tol = 1e-3;
	
	static int crack_exists = NO;
	static double pos = 1e10;
	static double pos0 = 1.;
	
	int i;
	int n = glob_mesh.size;
	if (crack_exists==NO){
		for (i=0;i<n;i++) if (H[i]<tol && glob_mesh.Points[i].y<pos){
			pos =  glob_mesh.Points[i].y;
			crack_exists = YES;
		}
		pos0 = pos;
	}
	else{
		for (i=0;i<n;i++) if (H[i]<tol && glob_mesh.Points[i].y<pos) pos =  glob_mesh.Points[i].y;
	}
	
	return (crack_exists) ? pos0-pos : 0;
}

double* compute_strain_energy(double* U,double* H,double* C,double* T,int order_dz){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double* E = elastic_potential(U,H,C,T,order_dz,PARTIAL0);
	double* Self = elastic_potential_self(U,H,C,T,order_dz,PARTIAL0,PARTIAL0);
	vector_add(E,Self,1.,n);
	
	vector_pseudo_div(E,V,n);
	free(V);
	free(Self);
	return E;
	
	/*double* X = zero_vector(n);
	multifrontal_solver(ID_damage,X,E,"compute_strain_energy",NULL);
	free(Self);
	free(E);
	return X;*/
}

double Free_energy_elastic(double* U,double* H,double* C,double* T){
	int n = glob_mesh.size;
	
	//double res = sparse_bilinear(U,Regularization,U)/2.;						// regularization
	double res = 0;
	double* F = elastic_potential(U,H,C,T,PARTIAL0,PARTIAL0);
	res += vector_contraction(F,n);												// elastic potential;
	
	double* Self = elastic_potential_self(U,H,C,T,PARTIAL0,PARTIAL0,PARTIAL0);
	res += vector_contraction(Self,n);											// self strain
	
	free(Self);
	free(F);
	return res;
}

double Free_energy_damage(double* H,double* C){
	int i;
	int n = glob_mesh.size;
	
	//double* Kappa = interface_energy_grad(C);
	//double* Res = sparse3D_bilinear(T_Flux,H,H);
	double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
	double* Grad = sparse3D_bilinear_tensor_2D_right(Strain_grad,H,H);
	double* Res = nodewise_tensor_contract_2D(Kappa,Grad);
	
	double res = vector_contraction(Res,n)/2.;
	//double res = kappa*scalar(Kappa,Res,n)/2.;
	//double res = sparse_bilinear(H,Interface_damage,H)/2.;					// interface energy
	
	double* Xi = interface_energy_pot(C);
	double* Fsqr = zero_vector(n);											// damage potential
	for (i=0;i<n;i++) Fsqr[i] = H[i]-1.;								
	double* Pot = sparse3D_bilinear(T_Source,Fsqr,Fsqr);
	res += xi*scalar(Xi,Pot,n);
	
	free(Kappa);
	free(Grad);
	free(Res);
	free(Xi);
	free(Pot);
	free(Fsqr);
	return res;
}

double Free_energy_chemical(double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	double res = sparse_bilinear(C,Interface_chemical,C)*kappa_c/2.;					// interface energy
	
		
	if (T==NULL){								
		double* Fsqr = zero_vector(n);													// Cahn-Hilliard potential
		for (i=0;i<n;i++) Fsqr[i] = C[i]*C[i]-1.;											
		res += sparse_bilinear(Fsqr,ID_damage,Fsqr);
		free(Fsqr);
	}
	else {
		double* F = free_energy_self_c(C,T);											// Landau potential
		res += scalar(V,F,n);
		free(F);
	}
	
	free(V);						
	return res;
}

double Total_free_energy(double* U,double* H,double* C,double* T){
	return Free_energy_elastic(U,H,C,T)+Free_energy_damage(H,C)+Free_energy_chemical(C,T)+Free_energy_heat(C,T);
}

double* get_constraint_force(double* U,double* H,double* H_prev,double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	
	double* Alpha = damage_rate_independent(C);
	//double* Kappa = interface_energy_grad(C);
	
	double* W = elastic_potential(U,H,C,T,PARTIAL1,PARTIAL0);
	double* W_self = elastic_potential_self(U,H,C,T,PARTIAL1,PARTIAL0,PARTIAL0);
		
	vector_add(W,W_self,1.,n);		
	
	double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
	double* K = sparse3D_times_rank2_times_scalar_2D_right(Strain_grad,Kappa,H);
	//scalar_mult(kappa,K,n);
	//double* K = sparse3D_vvB(T_Flux,Kappa,H,n);
	//double* K = sparse_mult(Interface_damage,H);
	double* S = damage_self_gradient(H,C,n);
	
	double* Res = sparse_mult(Dissipation,Alpha);
	vector_add(Res,S,-1.,n);
	vector_add(Res,W,-1.,n);
	vector_add(Res,K,-1.,n);
	
	double* dH = clone_vector(H,n);
	vector_add(dH,H_prev,-1.,n);
	linear_map(Res,-beta/dt,ID_damage,dH);
	
	//if (R_Newton_Condition!=NULL) for (i=0;i<n;i++) if (R_Newton_Condition[i]==DIRICHLET) Res[i] = H[i];		
	
	vector_pseudo_div(Res,V,n);
	
	free(Alpha);
	free(Kappa);
	free(W_self);
	free(V);
	free(W);
	free(K);
	free(S);
	free(dH);
	return Res;
}

double get_constraint_work(double* X){
	
	int n = glob_mesh.size;
	int n_prev = Glob_mesh_sizes[glob_time_index-1];
	
	double* U = &(X[0]);
	double* H = &(X[dimension*n]);
	double* C = &(X[(dimension+1)*n]);
	double* T = &(X[(dimension+3)*n]);
	double* H_prev = interpolated(All_H[glob_time_index-1],1,n_prev,NULL,NULL);
	
	double* V = set_vector_Ui_0_2D(1);
	double* P = get_constraint_force(U,H,H_prev,C,T);
	double* dH = clone_vector(H,n);
	vector_add(dH,H_prev,-1.,n); 
	vector_pseudo_mult(P,dH,n);
	double res = scalar(P,V,n);
	
	free(P);
	free(V);
	free(dH);
	free(H_prev);
	return res;	
}

double* damage_predictor(double* U,double* H,double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);	
	
	double* Alpha = damage_rate_independent(C);
	//double* Kappa = interface_energy_grad(C);
	double* W = elastic_potential(U,H,C,T,PARTIAL1,PARTIAL0);
	double* W_self = elastic_potential_self(U,H,C,T,PARTIAL1,PARTIAL0,PARTIAL0);
	vector_add(W,W_self,1.,n);		
	
	double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
	double* K = sparse3D_times_rank2_times_scalar_2D_right(Strain_grad,Kappa,H);	
	//double* K = sparse3D_vvB(T_Flux,Kappa,H,n);
	//scalar_mult(kappa,K,n);
	double* S = damage_self_gradient(H,C,n);
	
	double* Res = sparse_mult(Dissipation,Alpha);
	vector_add(Res,S,-1.,n);
	vector_add(Res,W,-1.,n);
	vector_add(Res,K,-1.,n);
	
	scalar_mult(1./beta,Res,n);
	vector_pseudo_div(Res,V,n);
	
	free(Alpha);
	free(Kappa);
	free(W_self);
	free(V);
	free(W);
	free(K);
	free(S);
	return Res;
}
																						
double* secondorder_predictor(double* U,double* H,double* C,double* T){
	int i;
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);	
	double* Alpha = damage_rate_independent(C);
	//double* Kappa = interface_energy_grad(C);
	
	double* W = elastic_potential(U,H,C,T,PARTIAL1,PARTIAL0);
	double* W_self = elastic_potential_self(U,H,C,T,PARTIAL1,PARTIAL0,PARTIAL0);
	vector_add(W,W_self,1.,n);			

	double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
	double* K = sparse3D_times_rank2_times_scalar_2D_right(Strain_grad,Kappa,H);
	//double* K = sparse3D_vvB(T_Flux,Kappa,H,n);
	//scalar_mult(kappa,K,n);
	
	double* Res = sparse_mult(Dissipation,Alpha);
	vector_add(Res,W,-1.,n);
	vector_add(Res,K,-1.,n);
	
	vector_pseudo_div(Res,V,n);
	
	double* R = clone_vector(H,n);
	
	
	double* Rate = generate_vector(n,beta/dt);
	double* Xi = interface_energy_pot(C);
	vector_add(Rate,Xi,2.*xi,n);
	
	vector_shift(R,n,-1.);
	vector_add(Res,R,beta/dt,n);
	vector_pseudo_div(Res,Rate,n);	
	
	free(Alpha);
	free(W_self);	
	free(Kappa);
	free(Rate);
	free(Xi);
	free(V);
	free(W);
	free(K);
	free(R);
	return Res;
}

double Damage_dissipation(double* H_prev,double* H,double* C){						// already multiplied by dt
	int i;
	int n = glob_mesh.size;
	double* Alpha = damage_rate_independent(C);
	double d = -sparse_bilinear(H,Dissipation,Alpha);
	if (H_prev==NULL){
		double* E = generate_vector(n,1.);
	
		d += sparse_bilinear(E,Dissipation,Alpha);
		free(E);
	}	
	else d += sparse_bilinear(H_prev,Dissipation,Alpha);
	
	double* V = clone_vector(H,n);
	if (H_prev==NULL){
		double* E = generate_vector(n,1.);
		vector_add(V,E,-1.,n);
		free(E);
	}
	else vector_add(V,H_prev,-1.,n);
	d += beta*sparse_bilinear(V,ID_damage,V)/dt;
	
	free(Alpha);
	free(V);
	return d;
}

double* Damage_dissipation_field(double* H_prev,double* H,double* C){						// already multiplied by dt
	int n = glob_mesh.size;
	double* Alpha = damage_rate_independent(C);
	double* dH = clone_vector(H,n);
	
	if (H_prev!=NULL) vector_add(dH,H_prev,-1.,n); else vector_shift(dH,n,-1.);
	double* Res = sparse_mult(Dissipation,dH);
	vector_pseudo_mult(Res,Alpha,n);	
	scalar_mult(-1.,Res,n);
	
	double* R = sparse3D_Bvv(T_Source,dH,dH);
	vector_add(Res,R,beta/dt,n);

	free(Alpha);
	free(dH);
	free(R);
	return Res;
}

double* Heat_source(double* U,double* U_prev,double* H,double* H_prev,double* C,double* C_prev,double* Mu,double* T){	
	int n = glob_mesh.size;
	double* Alpha = damage_rate_independent(C);

	double* dH = clone_vector(H,n);
	vector_add(dH,H_prev,-1.,n);
	
	double* dC = clone_vector(C,n);
	vector_add(dC,C_prev,-1.,n);
	
	double* Res = sparse3D_bilinear(T_Source,dH,dH);	
	scalar_mult(beta/dt,Res,n);
	double* D = sparse_mult(Dissipation,Alpha);
	vector_pseudo_mult(D,dH,n);
	vector_add(Res,D,-1.0,n);
	
	double* M = get_mobility(H,C,T);
	double* Chem = sparse3D_bilinear(T_Flux,Mu,Mu);
	vector_pseudo_mult(Chem,M,n);
	vector_add(Res,Chem,dt,n);
	double* Vis = sparse3D_bilinear(T_Source,dC,dC);
	vector_add(Res,Vis,vis_c/dt,n);
	
	free(Alpha);
	free(Vis);
	free(D);
	free(dH);
	free(dC);
	free(Chem);
	free(M);
	return Res;
}

double* vector_along_path(int* List,int size,double* X){
	int i,j,k,l,m;
	int n = glob_mesh.size;
	double* Res = zero_vector(dimension*n);
	for (i=0;i<size;i++){
		k = List[i];
		for (j=0;j<glob_mesh.Sizes[k];j++){
			l = glob_mesh.Connections[k][j];
			if (near_boundary(l)){
				for (m=0;m<dimension;m++) Res[l+m*n] = X[l+m*n];
			}			
		}
		for (m=0;m<dimension;m++) Res[k+m*n] = X[k+m*n];
	}
	return Res;
}

double Boundary_damage(double* H_prev,double* H){
	int k;
	int n = glob_mesh.size;	
	double res;
	double* dH = clone_vector(H,n);
	vector_add(dH,H_prev,-1.,n);

	set_var_number2D(2);	
	double* Kappa = damage_interface_tensor(H,Glob_Pointer_C,PARTIAL0);
	sparse_matrix3D* B = set_matrix_bound_2_Bijk_100_2D(0,0,0,&insert_B_abV_bV_aW);
	sparse_matrix* A = sparse3D_times_rank2_2D_left(B,Kappa);
	//sparse_matrix* B = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_bbV);
	sparse_matrix* Intbound = get_transpose(A,n);
	
	double* BD = zero_vector(n);
	linear_map(BD,(double)1/2,Intbound,H);
	linear_map(BD,(double)1/2,Intbound,H_prev);
	res = scalar(BD,dH,n);
	
	free_sparse3D(B);
	free_sparse(Intbound);
	free_sparse(A);
	free(Kappa);
	free(BD);
	free(dH);
	
	return res;
}

point2D Boundary_elastic_force(double* U_prev,double* U,double* H,double* C,double* Th_prev,double* Th){
	
	const int var = 0;
	const int loop = 0;
	const int segment = 1;
	const double tol = 1e-8;
			
	int i,j,k;
	double x,y,fx,fy;
	point2D res;
	
	int n = glob_mesh.size;
	//double d = (double)dimension;
	/*double* G = damage_weight_function(H);
	double* Dil = total_strain_trace(U,C,Th);
	
	double Ga = (mu_1+mu_2)/2.;
	double Gb = (mu_1-mu_2)/2.;
	double Ka = (lambda_1+lambda_2)/2.+2./d*Ga;
	double Kb = (lambda_1-lambda_2)/2.+2./d*Gb;	
	
	double* Shear = zero_vector(n);
	for (i=0;i<n;i++) Shear[i] = G[i]*(Ga+Gb*linear_limited(C[i]));
	double* Bulk = zero_vector(n);
	for (i=0;i<n;i++) Bulk[i] = (1.+damage_anisotropy(Dil[i])*(G[i]-1.))*(Ka+Kb*linear_limited(C[i]));
	
	set_var_number2D(dimension);
	sparse_matrix* Dirichlet = sparse3D_Bv(Bound_Dirichlet_shear,Shear);
	sparse_matrix* D2 = sparse3D_Bv(Bound_Dirichlet_bulk,Bulk);
	sparse_add(Dirichlet,D2,1.);
	sparse_add(Dirichlet,Bound_Dirichlet_reg,1.);
	sparse_matrix* DT = get_transpose(Dirichlet,dimension*n);
	
	free_sparse(Dirichlet);
	free_sparse(D2);
	free(Bulk);
	free(Shear);
	free(Dil);
	free(G);*/
	
	double* Rank4_modulus = elastic_constitutive(U,H,C,Th,PARTIAL0,PARTIAL0);
	sparse_matrix* DT = sparse3D_times_rank4_2D_right_T(Strain_grad_bound,Rank4_modulus);
	sparse_add(DT,Bound_Dirichlet_reg,1.);
	
	double* Dirichlet_part = zero_vector(dimension*n);
	linear_map(Dirichlet_part,(double)1/2,DT,U);
	linear_map(Dirichlet_part,(double)1/2,DT,U_prev);
	
	double* S0 = self_strain(C,Th_prev,PARTIAL0,PARTIAL0);
	double* S1 = self_strain(C,Th,PARTIAL0,PARTIAL0);
	double* T0 = linear_rank2(Rank4_modulus,S0);
	double* T1 = linear_rank2(Rank4_modulus,S1);
	
	linear_map(Dirichlet_part,(double)-1/2,Bound_Active,T0);
	linear_map(Dirichlet_part,(double)-1/2,Bound_Active,T1);
	
	free_sparse(DT);
	free(Rank4_modulus);
	free(S0);
	free(S1);
	free(T0);
	free(T1);
	
	int* Cond = get_segment_conditions(var,loop,segment);
	
	fx = 0;
	fy = 0;
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		if (Cond[i]==DIRICHLET || Cond[i]==ROBIN){
		//if (fabs(x-0.5)<tol || fabs(x+0.5)<tol){
			fx += Dirichlet_part[i];
			fy += Dirichlet_part[n+i];
		}
	}
	res = init_point2D(fx,fy);
	
	free(Cond);
	free(Dirichlet_part);
	return res;
}

point2D mean_bound_displacement(double* U_prev,double* U){
	
	const int var = 0;
	const int loop = 0;
	const int segment = 1;
	const double tol = 1e-8;
	
	int i;
	double x,y,ux,uy;
	point2D res;
	
	int n = glob_mesh.size;
	int* Cond = get_segment_conditions(var,loop,segment);
	int m = 0;
	ux = 0;
	uy = 0;
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		if (Cond[i]==DIRICHLET || Cond[i]==ROBIN){
		//if (fabs(x-0.5)<tol || fabs(x+0.5)<tol){
			ux += U_prev[i]+U[i];
			uy += U_prev[n+i]+U[n+i];
			m++;
		}
	}
	res = init_point2D((double)ux/(2*m),(double)uy/(2*m));
	
	free(Cond);
	return res;
}
	

double Boundary_elastic(double* U_prev,double* U,double* H,double* C,double* Th_prev,double* Th){		// Attention: add work from inner forces
	int i,j,k;
	double res,s;
	
	int n = glob_mesh.size;
	double* Rank4_modulus = elastic_constitutive(U,H,C,Th,PARTIAL0,PARTIAL0);
	sparse_matrix* DT = sparse3D_times_rank4_2D_right_T(Strain_grad_bound,Rank4_modulus);
	sparse_add(DT,Bound_Dirichlet_reg,1.);
	
/*#else
	// Es fehlt die damage Anisotropy !
	double h;
	double* F1 = zero_vector(n);
	double* F2 = zero_vector(n);
	for (i=0;i<n;i++){
		h = linear_limited(C[i]);
		F1[i] = G[i]*(1.+h)/2.;
		F2[i] = G[i]*(1.-h)/2.;
	}
	
	sparse_matrix* D1 = sparse3D_Bv(Bound_Dirichlet_1,F1);
	sparse_matrix* D2 = sparse3D_Bv(Bound_Dirichlet_2,F2);
	sparse_add(D1,D2,1.);
	sparse_add(D1,Bound_Dirichlet_reg,1.);
	sparse_matrix* DT = get_transpose(D1,dimension*n);
	
	
	free_sparse(D1);
	free_sparse(D2);
	free(F1);
	free(F2);
	
#endif*/
	
	double* Dirichlet_part = zero_vector(dimension*n);
	linear_map(Dirichlet_part,(double)1/2,DT,U);
	linear_map(Dirichlet_part,(double)1/2,DT,U_prev);
	
	double* S0 = self_strain(C,Th_prev,PARTIAL0,PARTIAL0);
	double* S1 = self_strain(C,Th,PARTIAL0,PARTIAL0);
	double* T0 = linear_rank2(Rank4_modulus,S0);
	double* T1 = linear_rank2(Rank4_modulus,S1);	
		
	linear_map(Dirichlet_part,(double)-1/2,Bound_Active,T0);
	linear_map(Dirichlet_part,(double)-1/2,Bound_Active,T1);
	
	bound_cond* Cond0 = get_U_bound_conditions(glob_time-dt);
	bound_cond* Cond1 = get_U_bound_conditions(glob_time);
	
	double* Neumann_part = zero_vector(dimension*n);
	for (i=0;i<dimension*n;i++){
		if (Cond1->Cond[i]==NEUMANN) Neumann_part[i] = (Cond1->Val[i]+Cond0->Val[i])/2.;		
	}
	sparse_multiplication(Bound_Neumann,Neumann_part);
	
	double* Robin_part = zero_vector(dimension*n);
	for (i=0;i<dimension*n;i++) if (Cond1->Cond[i]==ROBIN){
		s = (i/n==0) ? Robin_stiffness.x : Robin_stiffness.y;
		Robin_part[i] = s*((Cond1->Val[i]+Cond0->Val[i])/2.-(U[i]+U_prev[i])/2.);		
	}
	sparse_multiplication(Bound_Neumann,Robin_part);
	
	double* Strain = zero_vector(dimension*n);
	for (i=0;i<dimension*n;i++){
		if (Cond1->Cond[i]==DIRICHLET) Strain[i] = Dirichlet_part[i];
		if (Cond1->Cond[i]==NEUMANN) Strain[i] = Neumann_part[i];		
		if (Cond1->Cond[i]==ROBIN) Strain[i] = Robin_part[i];		
	}
	double* dU = clone_vector(U,dimension*n);
	vector_add(dU,U_prev,-1.,dimension*n);
	res = scalar(Strain,dU,dimension*n);
	
	free_bound_cond(&Cond0);
	free_bound_cond(&Cond1);
	free(Rank4_modulus);
	free_sparse(DT);
	free(S0);
	free(S1);
	free(T0);
	free(T1);
	free(dU);
	free(Strain);
	free(Robin_part);
	free(Dirichlet_part);
	free(Neumann_part);	
	return res;
}

void UMFPACK_settings(){
	// pattern
	
	// factorization
	
	// solve
}

sparse_matrix* get_CH_linear_part(double* H,double* Y, double* T){
	int n = glob_mesh.size;
	double* C = &(Y[0]);
	
	// first equation
	sparse_matrix* A = clone(Phase_ID_c);
	//sparse_add(A,Phase_Diffusion_mu,mobility*dt);
	double* M = get_mobility(H,C,T);
	sparse_matrix* Grad2 = sparse3D_vB(Phase_Diffusion_mu,M,2*n);
	sparse_add(A,Grad2,dt);
	free_sparse(Grad2);
	free(M);
	
	// second equation	
	sparse_add(A,Phase_ID_mu,1.);
	sparse_add(A,Phase_Diffusion_c,-kappa_c);
	sparse_add(A,Phase_ID_cross,-vis_c/dt);
	
	return A;
}

double* Thermo_dF(double* H,double* H_prev,double* C,double* C_prev,double* U,double* U_prev,double* T){
	int i;
	int n = glob_mesh.size; 
	double* V = set_vector_Ui_0_2D(1);
	
	double* Res = zero_vector(n);
	double* M = chemical_potential_dT(U,H,C,T);												// T*d/dt(mu)*dc
	for (i=0;i<n;i++) Res[i] += M[i]*(C[i]-C_prev[i]);				
	free(M);
	
	double* S = sigma_dT(U,H,C,T);															// T*d/dt(sigma):de
	double* dU = clone_vector(U,dimension*n);
	vector_add(dU,U_prev,-1.,dimension*n);
	double* dE = linear_strain_tensor(dU);
	for (i=0;i<n;i++) Res[i] += (S[i]*dE[i]+S[n+i]*dE[n+i]+S[2*n+i]*dE[2*n+i]+S[3*n+i]*dE[3*n+i])/V[i];
	free(S);
	free(dU);
	free(dE);
	
	double* G = damage_potential_dT(U,H,C,T);												// T*d/dt(gamma):dz
	for (i=0;i<n;i++) Res[i] += G[i]*(T[i]+T_absolute_ref)*(H[i]-H_prev[i]);	
	free(G);
	
	vector_pseudo_div(Res,V,n);
	free(V);
	return Res;
}

void temp_exp_mult(double* X,double* H,double* H_prev,double* C,double* C_prev,double* U,double* U_prev,double* T){
	int i;
	int flag = 0;
	int n = glob_mesh.size;
	double* CV = heat_capacity_field(U,H,C,T);
	double* dF = Thermo_dF(H,H_prev,C,C_prev,U,U_prev,T);
	vector_pseudo_div(dF,CV,n);
	for (i=0;i<n;i++){
		X[i] *= exp(dF[i]);
		if ((dF[i]>1.) && flag==0){
			printf("Warning: something nasty happens with heat transport at node index %d\n",i);
			flag = 1;
		}
	}
	free(CV);
	free(dF);
}

sparse_matrix* get_Tdyn_left_side_matrix(double* H,double* H_prev,double* C,double* C_prev,double* U,double* U_prev,double* T){
	int i;
	double c;
	
	int n = glob_mesh.size;
	double d = (double)dimension;	
	double ka = (kappa_T1+kappa_T2)/2.-kappa_T0;
	double kb = (kappa_T1-kappa_T2)/2.;
	
	double* K = zero_vector(n);
	for (i=0;i<n;i++){
		c = linear_limited(C[i]);
		K[i] = kappa_T0+(ka+kb*c)*H[i];
	}
	
	
#ifdef TEMP_EQUILIB
	sparse_matrix* DT = sparse_zero(n);
#else
	sparse_matrix* DT = heat_capacity(U,H,C,T);									// c*dT/dt
#endif	

	sparse_matrix* Flux = sparse3D_vB(T_Flux,K,n);								// -div(K*gradT)
	sparse_add(DT,Flux,dt);
	
	free_sparse(Flux);
	free(K);
	return DT;
}

double* get_Tdyn_right_side_vector(double* H,double* H_prev,double* C,double* C_prev,double* Y,
  double* U,double* U_prev,double* T,double* T_prev){
	int i;
	
	int n = glob_mesh.size;
	double d = (double)dimension;
	double ra = (rho_1+rho_2)/2.;
	double rb = (rho_1-rho_2)/2.;	
	double ka = (kappa_T1+kappa_T2)/2.-kappa_T0;
	double kb = (kappa_T1-kappa_T2)/2.;
	double* V = set_vector_Ui_0_2D(1);
	double* R = zero_vector(n);
	//double* N = zero_vector(n);	
	
#ifdef TEMP_EQUILIB
	sparse_matrix* DT = sparse_zero(n);
#else
	sparse_matrix* DT = heat_capacity(U,H,C,T);									// c*dT/dt, -T*d^2/dT^2(W_el)  neglected
#endif	

	double* Res = sparse_mult(DT,T_prev);
		
	//linear_map(Res,dt,T_Neumann,N);			
	free_sparse(DT);
	//free(N);						

	double* Q = Heat_source(U,U_prev,H,H_prev,C,C_prev,Y,T);
	vector_add(Res,Q,1.,n);
	free(Q);
	
	double* M = chemical_potential_dT(U,H,C,T);												// T*d/dt(mu)*dc
	for (i=0;i<n;i++) Res[i] += M[i]*(T[i]+T_absolute_ref)*(C[i]-C_prev[i]);				
	free(M);
	
	double* S = sigma_dT(U,H,C,T);															// T*d/dt(sigma):de
	double* dU = clone_vector(U,dimension*n);
	vector_add(dU,U_prev,-1.,dimension*n);
	double* dE = linear_strain_tensor(dU);
	for (i=0;i<n;i++) Res[i] += (T[i]+T_absolute_ref)*(S[i]*dE[i]+S[n+i]*dE[n+i]+S[2*n+i]*dE[2*n+i]+S[3*n+i]*dE[3*n+i])/V[i];
	free(S);
	free(dU);
	free(dE);
	
	double* G = damage_potential_dT(U,H,C,T);												// T*d/dt(gamma):dz
	for (i=0;i<n;i++) Res[i] += G[i]*(T[i]+T_absolute_ref)*(H[i]-H_prev[i]);	
	free(G);
	
	free(V);
	return Res;
}

void Tdyn_apply_BC(sparse_matrix* A,double* b,bound_cond* Tcond){
	int i;
	
	int n = glob_mesh.size;
	double* N = zero_vector(n);
	for (i=0;i<n;i++) if (Tcond->Cond[i]==NEUMANN || Tcond->Cond[i]==ROBIN) N[i] = Tcond->Val[i];
	linear_map(b,dt,T_Neumann,N);		
	Set_dirichlet_BC(A,b,Tcond);
	free(N);
}

double* interpolated(double* X,int dim,int size_per_dim,sparse_matrix* LS_Matrix,sparse_matrix* RS_Matrix){
	const int iter = -1000;
	const double relax = 1.0;
	
	int status;
	sparse_matrix* Inter = NULL;
	double* Res = NULL;
	
	if (space_adaptive==NO) return clone_vector(X,dim*size_per_dim);
	if (RS_Matrix==NULL) Inter = Interpolation; else Inter = RS_Matrix;
	if (Inter!=NULL){	
		int i;
		double* B;
		double* Z;
		sparse_matrix* A;
		int new_size = Inter->size;
		set_var_number2D(1);
		if (interpolation_method==ELEMENT_BASE){
			if (LS_Matrix==NULL) A = set_matrix_Aij_00_2D(0,0,&insert_AV); else A = LS_Matrix;
		}
		else A = sparse_identity(Inter->size);
		Res = zero_vector(dim*new_size);
		for (i=0;i<dim;i++){
			Z = zero_vector(new_size);
			B = sparse_mult(Inter,&(X[i*size_per_dim]));
			//SOR(A,B,Z,relax,iter);			
			status = multifrontal_solver(A,Z,B,"interpolated",&UMFPACK_time_interpolation);			
			
			copy_vector_content(Z,Res,0,i*new_size,new_size);
			free(Z);
			free(B);
		}
		if (LS_Matrix==NULL){free_sparse(A);}
	}
	else{
		printf("interpolated: no interpolation matrix set -> abort\n");
		exit(0);
	}
	return Res;
}

void set_matrices(){
	double d = (double)dimension;
	int n = glob_mesh.size;
	if (dimension==2){
		
		// elastic energy
		
		set_var_number2D(1);
		Strain_grad = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_V_aB_abV_b);
		Strain_abs = set_matrix_Bijk_001(n_h,n_h,n_u,&insert_UBa_V_a);
		set_var_number2D(dimension);
		sparse_matrix3D* Strain_grad_bound_T = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_baV_bV_aW);	
		Strain_grad_bound = get_transpose3D_13(Strain_grad_bound_T,n);
		free_sparse3D(Strain_grad_bound_T);
		
	#ifndef ANISOTROPIC_ELASTICITY
		/*set_var_number2D(1);
		Elastic_bulk = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_V_aB_abV_b);
		Elastic_shear = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_V_aB_baV_b);		
		sparse_matrix3D* B3 = set_matrix_Bijk_011(n_h,n_u,n_u,&insert_B_aaV_bV_b);
		sparse3D_add(Elastic_shear,B3);
		free_sparse3D(B3);
		sparse3D_scalar_mult(Elastic_bulk,-2./d);
		sparse3D_add(Elastic_shear,Elastic_bulk);
		sparse3D_scalar_mult(Elastic_bulk,-d/2.);
		sparse3D_merge(Elastic_shear);*/
	
	#else
		/*set_var_number2D(1);
		set_moduli_phase(1,TRACELESS);
		Elastic_trl_1 = set_matrix_Bijk_011(0,0,0,&insert_moduli);
		set_moduli_phase(1,DEVIATOR);
		Elastic_dev_1 = set_matrix_Bijk_011(0,0,0,&insert_moduli);
		set_moduli_phase(2,TRACELESS);
		Elastic_trl_2 = set_matrix_Bijk_011(0,0,0,&insert_moduli);
		set_moduli_phase(2,DEVIATOR);
		Elastic_dev_2 = set_matrix_Bijk_011(0,0,0,&insert_moduli);*/
	#endif
		// regularization
		set_var_number2D(dimension);
		Regularization = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_bbV_a);	
	#ifndef ANISOTROPIC_ELASTICITY
		double mu = (mu_1<mu_2) ? mu_1 : mu_2;
	#else
		double mu = (Moduli_1.reg<Moduli_2.reg) ? Moduli_1.reg : Moduli_2.reg;
	#endif
		scalar_sparse_mult(mu*eta_eps,Regularization);
		
		// boundary matrices
		sparse_matrix* Bound_Dirichlet_reg_T = set_matrix_bound_2_Aij_10_2D(n_u,n_u,&insert_A_bbV_a);
		Bound_Dirichlet_reg = get_transpose(Bound_Dirichlet_reg_T,dimension*n);
		scalar_sparse_mult(mu*eta_eps,Bound_Dirichlet_reg);
		free_sparse(Bound_Dirichlet_reg_T);
		
		Bound_Neumann_reg = set_matrix_bound_0_Aij_00_2D(n_u,n_u,&insert_AV_a);
		scalar_sparse_mult(mu,Bound_Neumann_reg);
		
	#ifndef ANISOTROPIC_ELASTICITY
		/*Bound_Dirichlet_bulk = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_abV_bV_aW);	
		Bound_Dirichlet_shear = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_baV_bV_aW);
		sparse_matrix3D* Bound_B3 = set_matrix_bound_2_Bijk_100_2D(n_u,n_u,n_h,&insert_B_aaV_bV_bW);	
		sparse3D_add(Bound_Dirichlet_shear,Bound_B3);
		free_sparse3D(Bound_B3);
		sparse3D_scalar_mult(Bound_Dirichlet_bulk,-d/2.);
		sparse3D_add(Bound_Dirichlet_shear,Bound_Dirichlet_bulk);
		sparse3D_scalar_mult(Bound_Dirichlet_bulk,-2./d);
		sparse3D_merge(Bound_Dirichlet_shear);*/
	#else 
		/*set_moduli_phase(1,TRACELESS);
		Bound_Dirichlet_trl_1 = set_matrix_bound_2_Bijk_100_2D(0,0,0,&insert_moduli_bound);
		set_moduli_phase(1,DEVIATOR);
		Bound_Dirichlet_dev_1 = set_matrix_bound_2_Bijk_100_2D(0,0,0,&insert_moduli_bound);
		set_moduli_phase(2,TRACELESS);
		Bound_Dirichlet_trl_2 = set_matrix_bound_2_Bijk_100_2D(0,0,0,&insert_moduli_bound);
		set_moduli_phase(2,DEVIATOR);
		Bound_Dirichlet_dev_2 = set_matrix_bound_2_Bijk_100_2D(0,0,0,&insert_moduli_bound);	*/
	#endif
		
		// linear strain
		Strain_tensor_Ax = set_matrix_Aij_01_2D(n_u,n_u,&insert_A_aV_b_row_x);
		Strain_tensor_Ay = set_matrix_Aij_01_2D(n_u,n_u,&insert_A_aV_b_row_y);
		
		Bound_Neumann = set_matrix_bound_0_Aij_00_2D(n_u,n_u,&insert_AV_a);
		Bound_Active = set_matrix_bound_1_Aij_00_2D(n_u,n_u,&insert_A_bV_ba);
		
		Jacobian_Ux = set_matrix_Bijk_001(n_h,n_h,n_u,&insert_B_aWV_b_rowx);
		Jacobian_Uy = set_matrix_Bijk_001(n_h,n_h,n_u,&insert_B_aWV_b_rowy);
		
		// interface energy
		set_var_number2D(1);
		//Interface_damage = set_matrix_Aij_11_2D(n_h,n_h,&insert_A_bbV);		
		//print_to_matrix_market_format(Interface_damage,"/Home/damage/radszuwe/Daten/extralarge_sparse");		
		
		//scalar_sparse_mult(kappa,Interface_damage);
		//Interface_damage = NULL;
		
		// damage treshold
		Dissipation = set_matrix_Aij_00_2D(n_h,n_h,&insert_AV);	
		
		// local damage potential
		ID_damage = set_matrix_Aij_00_2D(n_h,n_h,&insert_AV);	
		
		// damage Hesse
		//Damage_Hesse_linear = clone(ID_damage);										// self part xi*(z-1)^2
		//scalar_sparse_mult(2.*xi*dt,Damage_Hesse_linear);	
		//sparse_add(Damage_Hesse_linear,Interface_damage,dt);						// rate independent part
		//sparse_add(Damage_Hesse_linear,ID_damage,beta);								// rate dependent part
		
		// phasefield
		Interface_chemical = set_matrix_Aij_11_2D(n_c,n_c,&insert_A_bbV);
		set_var_number2D(dimension);
		Phase_Jacobian = set_matrix_Bijk_000(n_m,n_c,n_c,&insert_BWWW);
		Phase_Diffusion_c = set_matrix_Aij_11_2D(n_m,n_c,&insert_A_bbV);
		
		
		//Phase_Diffusion_mu = set_matrix_Aij_11_2D(n_c,n_m,&insert_A_bbV);
		Phase_Diffusion_mu = set_matrix_Bijk_011(0,n_c,n_m,&insert_B_aaWWW);
		Phase_ID_c = set_matrix_Aij_00_2D(n_c,n_c,&insert_AV);
		Phase_ID_mu = set_matrix_Aij_00_2D(n_m,n_m,&insert_AV);
		Phase_ID_cross = set_matrix_Aij_00_2D(n_m,n_c,&insert_AV);
		sparse3D_merge(Phase_Jacobian);
	
		// temperature
		set_var_number2D(1);
		//T_divu = set_matrix_Bijk_001(n_T,n_T,n_u,&insert_B_aWWV_a);
		T_Source = set_matrix_Bijk_000(n_T,n_T,n_T,&insert_BWWW);
		T_Flux = set_matrix_Bijk_011(n_h,n_T,n_T,&insert_B_aaWWW);
		T_Flux_Reg = set_matrix_Aij_11_2D(n_T,n_T,&insert_A_bbV);
		T_Capacity = set_matrix_Aij_00_2D(n_T,n_T,&insert_AV);
		T_Neumann = set_matrix_bound_0_Aij_00_2D(n_T,n_T,&insert_AV);		
	}
	if (dimension==3){
		printf("dimension 3 not implemented yet ...");
		exit(0);
	}
}

void set_glob_mesh(mesh2D* Mesh,element_collection* Triangles){
	if (Mesh!=NULL && Triangles!=NULL){
		glob_mesh.size = Mesh->size;
		glob_mesh.Sizes = Mesh->Sizes;
		glob_mesh.Connections = Mesh->Connections;
		glob_mesh.Points = Mesh->Points;
		glob_mesh.Is_boundary = Mesh->Is_boundary;
		glob_mesh.Overlap_table = Mesh->Overlap_table;
		
		elements.size = Triangles->size;
		elements.Elements = Triangles->Elements;
	}
}

void search_for_load_dir(char* Name,char* Param_name){
	const int num = 2;
	const char* Dir[] = {"/Home/damage/radszuwe/Daten","/scratch/radszuwe/Daten"};
	
	int i;
	FILE* file;
	int Found[num];
	char Buffer[512];
	
	int sum = 0;	
	for (i=0;i<num;i++){
		Found[i] = 0;
		sprintf(Buffer,"%s/%s/%s",Dir[i],Name,Default_param_name);
		//printf("%s/%s/%s\n",Dir[i],Name,Default_param_name);
		file = fopen(Buffer,"r");
		if (file!=NULL){
			Found[i] = 1;
			sum++;
			fclose(file);
		}
		//else printf("error code: %d\n",errno);
	}
	
	if (sum==0){
		printf("could not find file %s in search directories -> abort\n",Name);
		exit(0);		
	}
	
	if (sum==1){
		for (i=0;i<num;i++) if (Found[i]) sprintf(Param_name,"%s/%s/%s",Dir[i],Name,Default_param_name);
		return;
	}
	
	if (sum>1){
INPUT_DIR:
		printf("\n multiple files found: choose directory:\n\n");
		for (i=0;i<num;i++) if (Found[i]) printf("%s : (%d)\n",Dir[i],i);
		Buffer[0] = (char)getchar();
		Buffer[1] = '\0';
		i = atoi(Buffer);
		if (i>=num){
			printf("invalid option\n");
			goto INPUT_DIR;
		}			
		sprintf(Param_name,"%s/%s/%s",Dir[i],Name,Default_param_name);
		printf("selected: %s\n -> continue ...\n",Param_name);		
	}
}

void setup(int argc, char* argv[],double** Initial, void (*Init_func)(double* X)){
	
	int i,size,len;
	FILE* Output_global = NULL;
	FILE* Output_local = NULL;
	char Param_filename[512];
	char Fullname[512];
	char Command[512];
	char Buffer[512];
	char Shortname[512];
	
	char** Parts;
	char** Param_names;
	char** Param_vals;	
	
#ifdef CONSTANT_PHASE
	interpolation_method = BARYCENTRIC;
#endif

#ifdef FORCE_ELEMENT_BASED_INTERPOLATION
	interpolation_method = ELEMENT_BASE;
#endif
	
	// read parameters
	double tstart = 0;
	char* To_load = NULL;
	time_offset = DBL_MIN;
	Parts = split(argv[2],"=",&len);
	if (len==2 && strcmp(Parts[0],"-load")==0){
		int slen = 0;
		char** Sub = split(Parts[1],"*",&slen);
		if (slen==2) sprintf(Param_filename,"%s/%s",Param_dir,Sub[1]);		
		else sprintf(Param_filename,"%s/%s/%s",Home_dir,Parts[1],Default_param_name);		
		
		
		FILE* file = fopen(Param_filename,"r");
		if (file!=NULL) fclose(file); else search_for_load_dir(Parts[1],Param_filename); 	
		
		To_load = (char*)malloc(512*sizeof(char));
		if (slen<2) strcpy(To_load,Parts[1]); else strcpy(To_load,Sub[0]);
		
		for (i=0;i<slen;i++) free(Sub[i]);
		free(Sub);
	}
	else sprintf(Param_filename,"%s/%s",Param_dir,argv[2]);
	for (i=0;i<len;i++) free(Parts[i]);
	free(Parts);
	
	len = 0;
	if (!Read_param_strings(Param_filename,&Param_names,&Param_vals,&len)){
		printf("Invalid parameter file %s-> abort\n",Param_filename);
		exit(0);
	}
	else printf("Read parameter file: %s\n",Param_filename);
	Assign_param_strings(Param_names,Param_vals,len);
	
	if (To_load!=NULL){
		if (strstr(To_load,"/")!=NULL){
			char* Last_part = strrchr(To_load,'/')+1;		
			strcpy(Load_name,Last_part);
		}
		else strcpy(Load_name,To_load);
		free(To_load);
	}
	
	// read console parameters
	Read_console_parameters(argc,argv);
	
	// adjust time step if necessary
	if (roundl((long double)total_time/dt)!=time_steps){
		long double ldt = (long double)total_time/time_steps;
		dt = (double)ldt;
		printf("Warning: time step does not match step number -> adjust dt=%e\n",dt);
	}
	
	// create main output directory
	sprintf(Fullname,"%s/%s",Output_dir,Output_name);
	if (mkdir(Fullname,0777)==-1){
		printf("File %s already exists -> abort\n",Fullname);
		exit(0);
	}
	
	// create mesh directory
	sprintf(Fullname,"%s/%s/meshes",Output_dir,Output_name);
	if (mkdir(Fullname,0777)==-1){
		printf("Could not create mesh directory in %s -> abort\n",Fullname);
		exit(0);
	}
	
	// load initial data if set	
	*Initial = NULL;
	if (strcmp(Load_name,LOAD_NONE)!=0) load_mode = YES;
	if (load_mode==YES){
		sprintf(Fullname,"%s/%s/%s",Output_dir,Load_name,Default_param_name);
		char* Mdir = Get_parameter(Fullname,P_MESHDIR);
		char* Mnam = Get_parameter(Fullname,P_MESHNAME);
		if (Mdir!=NULL && Mnam!=NULL){
			sprintf(Fullname,"%s/%s",Output_dir,Load_name);
			if (binary_load(Fullname,Initial,&tstart,&size,load_index)==SUCCESS){						
				sprintf(Mesh_dir,"%s/%s",Output_dir,Load_name);
				sprintf(Mesh_name,"%s",Mnam);											
			}
			else printf("could not read initial data from file %s -> skip\n",Fullname);
		}
		else printf("could not load mesh from file %s/%s -> skip\n",Mdir,Mnam);		
		
		int load_index = glob_mesh_index;
		Parts = split(Mesh_name,".",&len);
		sprintf(Command,"cp %s/%s/meshes/%s.%d.node %s/%s/meshes/%s.1.node",Output_dir,Load_name,Parts[0],load_index,Output_dir,Output_name,Parts[0]);
		system(Command);
		sprintf(Command,"cp %s/%s/meshes/%s.%d.ele %s/%s/meshes/%s.1.ele",Output_dir,Load_name,Parts[0],load_index,Output_dir,Output_name,Parts[0]);
		system(Command);
		
		memcpy(Shortname,Mesh_name,strlen(Mesh_name)*(sizeof(Mesh_name)));
		char* SN = strtok(Shortname,".");
		sprintf(Command,"cp %s/%s.poly %s/%s",Mesh_dir,SN,Output_dir,Output_name);
		system(Command);	
		
		free(Mdir);
		free(Mnam);
	}	
	else{	
		Parts = split(Mesh_name,".",&len);
		sprintf(Command,"cp %s/%s.node %s/%s/meshes",Mesh_dir,Mesh_name,Output_dir,Output_name);
		system(Command);
		sprintf(Command,"cp %s/%s.ele %s/%s/meshes",Mesh_dir,Mesh_name,Output_dir,Output_name);
		system(Command);
		sprintf(Command,"cp %s/%s.poly %s/%s",Mesh_dir,Parts[0],Output_dir,Output_name);	
		system(Command);
	}	
	sprintf(Command,"cp %s/%s.node %s/%s",Mesh_dir,Mesh_name,Output_dir,Output_name);
	system(Command);
	sprintf(Command,"cp %s/%s.ele %s/%s",Mesh_dir,Mesh_name,Output_dir,Output_name);
	system(Command);
	
	
	
	if (len>1) glob_mesh_index = atoi(Parts[1]); else glob_mesh_index = 1;
	for (i=0;i<len;i++) free(Parts[i]);
	free(Parts);
	if (strcmp(Cond_name,"none")!=0){
		sprintf(Fullname,"%s/%s",Param_dir,Cond_name);
		
		int status = load_condition_file(Fullname,Boundary_conditions,degrees_of_freedom,&Initial_conditions);
		if (status==FAILED){
			printf("load condition file: no valid format or no existing file -> skip\n");
			sprintf(Cond_name,"none");
		}
	}
	else printf("no condition file given -> use default\n");
	
	//initial time
	if (time_offset==DBL_MIN) time_offset = tstart;	
	
	// write parameters
	Write_parameters(Default_param_name);

	// general settings
	//set_thread_num(omp_proc_num);
	UMFPACK_settings();
	
	// load base mesh
	read_mesh_2D(Mesh_dir,Mesh_name,&base_mesh,&base_elements,&Attributes,&attribute_num,CHATTY);	
	
	// set initial mesh
	if (load_mode==YES) Load_mesh(glob_mesh_index,&glob_mesh,&elements,QUIET);
	else{		
		set_glob_mesh(&base_mesh,&base_elements);
		if (space_adaptive==YES){
			glob_mesh_index--;
			double* Init_data = zero_vector(glob_mesh.size*degrees_of_freedom);
			double* Refine_data = generate_vector(glob_mesh.size,-1.);
			(*Init_func)(Init_data);
			refine_by_gradient(Init_data,Refine_data,degrees_of_freedom,min_refine_area,max_refine_area);
			
			create_refined_mesh(Refine_data,&(Init_data[dimension*glob_mesh.size]),NULL,QUIET);
			Load_mesh(glob_mesh_index,&glob_mesh,&elements,QUIET); 
			free(Init_data);
			free(Refine_data);
		}
		*Initial = zero_vector(glob_mesh.size*degrees_of_freedom);
		(*Init_func)(*Initial);
	}
	
	max_refine_area = get_largest_element_area(&base_mesh,&base_elements);
	
	// init data output
	init_binary_save(Bin_filename);
	Output_global = open_global();
	Output_local = open_local();
	if (Output_local==NULL || Output_global==NULL){
		printf("could not create output files -> abort\n");
		exit(0);
	}
	fclose(Output_global);
	fclose(Output_local);
			
	// set basic matrices
	set_matrices();
	
	// init statistics
	char Stat1[512];
	char Stat2[512];
	sprintf(Stat1,"%s/%s/%s",Output_dir,Output_name,"AMG_step1");
	step1_AMG_stat = AMG_init_stat(Fullname);
	
	sprintf(Stat1,"%s/%s/%s",Output_dir,Output_name,"AMG_step2");
	sprintf(Stat2,"%s/%s/%s",Output_dir,Output_name,"BiCGStab_step2");
	BI_set_stat(Stat1,Stat2);
	
	sprintf(Stat1,"%s/%s/%s",Output_dir,Output_name,"Altermin");
	alter_min_stat = AMG_init_stat(Stat1);
	
	RN_init_statistics();
}

void adjust_dt(double* X,double* Pred,double* dt,double dtmin,double dtmax,double constr_work){
	const int sample = 18;
	const int sub_iter = 10;
	
	//const int iter = 7;
	//const int thres = 10;
	
	static int counter = 0;
	static int max = 0;
	//static int Iter_history[20];		// must hold sub_iter*iter elements
	
	int i,flag;
	double old_dt;
	if (Pred!=NULL){		
		*dt /= (double)sub_iter;		
		if (*dt<dtmin) *dt = dtmin;
		printf("\ndecrease timestep to %e\n",*dt);
		counter = 0;
		max = 0;
	}
	else{
		//if (constr_work>eps_constraint) counter = 0; else counter++;
		counter++;
		if (constr_work>max) max = constr_work;
		if (counter>=sample){
			if (max<eps_constraint/10.){
				old_dt = *dt;
				*dt *= (double)sub_iter;
				if (*dt>dtmax) *dt = dtmax;
				if (*dt>old_dt) printf("\nincrease timestep to %e\n",*dt);
			}
			counter = 0;
			max = 0;
		}
		
	}	
	
}

void clean_matrices(){
	
	if (Strain_grad!=NULL){
		free_sparse3D(Strain_grad);
		Strain_grad = NULL;
	}
	if (Strain_abs!=NULL){
		free_sparse3D(Strain_abs);
		Strain_abs = NULL;
	}	
	if (Phase_Jacobian!=NULL){
		free_sparse3D(Phase_Jacobian);
		Phase_Jacobian = NULL;
	}
	if (Strain_grad_bound!=NULL){
		free_sparse3D(Strain_grad_bound);
		Strain_grad_bound = NULL;
	}	
	if (T_Flux!=NULL){
		free_sparse3D(T_Flux);
		T_Flux = NULL;
	}
	
	if (T_Source!=NULL){
		free_sparse3D(T_Source);
		T_Source = NULL;
	}
	
	if (Jacobian_Ux!=NULL){
		free_sparse3D(Jacobian_Ux);
		Jacobian_Ux = NULL;
	}
	
	if (Jacobian_Uy!=NULL){
		free_sparse3D(Jacobian_Uy);
		Jacobian_Uy = NULL;
	}
	if (Regularization!=NULL){
		free_sparse(Regularization);
		Regularization = NULL;
	}
	/*if (Interface_damage!=NULL){
		free_sparse(Interface_damage);
		Interface_damage = NULL;
	}*/
	if (Interface_chemical!=NULL){
		free_sparse(Interface_chemical);
		Interface_chemical = NULL;
	}
	if (Dissipation!=NULL){
		free_sparse(Dissipation);	
		Dissipation = NULL;
	}
	if (Bound_Dirichlet_reg!=NULL){
		free_sparse(Bound_Dirichlet_reg);
		Bound_Dirichlet_reg = NULL;
	}
	if (Bound_Neumann_reg!=NULL){
		free_sparse(Bound_Neumann_reg);
		Bound_Neumann_reg = NULL;
	}
	if (Bound_Neumann!=NULL){
		free_sparse(Bound_Neumann);
		Bound_Neumann = NULL;
	}
	if (ID_damage!=NULL){
		free_sparse(ID_damage);
		ID_damage = NULL;
	}
	if (Phase_Diffusion_c!=NULL){
		free_sparse(Phase_Diffusion_c);
		Phase_Diffusion_c = NULL;
	}
	if (Phase_Diffusion_mu!=NULL){
		free_sparse3D(Phase_Diffusion_mu);
		Phase_Diffusion_mu = NULL;
	}
	if (Phase_ID_c!=NULL){
		free_sparse(Phase_ID_c);
		Phase_ID_c = NULL;
	}
	if (Phase_ID_mu!=NULL){
		free_sparse(Phase_ID_mu);
		Phase_ID_mu = NULL;
	}
	if (Phase_ID_cross!=NULL){
		free_sparse(Phase_ID_cross);
		Phase_ID_cross = NULL;
	}
	if (Bound_Active!=NULL){
		free_sparse(Bound_Active);
		Bound_Active = NULL;
	}
	if (T_Flux_Reg!=NULL){
		free_sparse(T_Flux_Reg);
		T_Flux_Reg = NULL;
	}
	if (T_Capacity!=NULL){
		free_sparse(T_Capacity);
		T_Capacity = NULL;
	}
	
	if (T_Neumann!=NULL){
		free_sparse(T_Neumann);
		T_Neumann = NULL;
	}
	
	if (Strain_tensor_Ax!=NULL){
		free_sparse(Strain_tensor_Ax);
		Strain_tensor_Ax = NULL;
	}
	
	if (Strain_tensor_Ay!=NULL){
		free_sparse(Strain_tensor_Ay);
		Strain_tensor_Ay = NULL;
	}
}

void clean(){
	int i;
	if (strcmp(Cond_name,"none")!=0) for (i=0;i<F_PRIM;i++){
		if (Boundary_conditions[i]!=NULL) free_bound_info(&(Boundary_conditions[i]));
	}
	FreeMesh(&base_mesh);
	free(base_elements.Elements);
	clean_matrices();
	free(Glob_mesh_sizes);
}

void shift(double* X,double factor,double* Lower,int n){
	int i;
	double d;
	for (i=0;i<n;i++){
		if (!R_Newton_Condition[i] && !isnan(Lower[i])) X[i] += factor*(X[i]-Lower[i]);
	}
}

void detect_lower_equal_upper(double* Lower,double* Upper,int* Cond,sparse_matrix* A,int n,double tol){
	int i;
	for (i=0;i<n;i++) if (Cond[i]!=DIRICHLET){
		if (fabs(Upper[i]-Lower[i])<tol){
			if (A!=NULL){
				reset_row(A,i);
				insert_sparse(A,1.,i,i);
			}
			Cond[i] = DIRICHLET;
			Lower[i] = NAN;
			Upper[i] = NAN;
		}
		else Cond[i] = NOCON;
	}
}

/*void AMG_setting_step1(amg_system_info* AMG_info){
	AMG_print_info(NO);
	AMG_set_matrix_tolerance(step1_tol);
	AMG_set_max_iter(step1_max_iter);
	AMG_set_eps(step1_eps);
	AMG_set_smoother(&AMG_SOR_smoother);
	AMG_set_SOR_relaxation_coeff(1.6);
	
	
}

void AMG_setting_step2(amg_system_info* AMG_info){
	AMG_print_info(YES);
	AMG_set_matrix_tolerance(step2_tol);
	AMG_set_max_iter(step2_max_iter);
	AMG_set_eps(step2_eps);
	AMG_set_smoother(&AMG_parallel_Jacobi_smoother);
	RN_set_AMG_data(AMG_info);
	BI_set_AMG_setup_data(AMG_info);
}

int AMG_Solve(sparse_matrix* A,double* X,double* b,int dim){
	
	AMG_print_info(YES);
	AMG_set_max_iter(step1_max_iter);
	AMG_set_matrix_tolerance(step1_tol);
	AMG_set_eps(step1_eps);
	AMG_set_smoother(&AMG_SOR_smoother);
	AMG_set_SOR_relaxation_coeff(1.6);
	int smooth_miter = 2;
	
	amg_system_info* Setup_data = AMG_setup(A,dim,step1_AMG_depth,smooth_miter,smooth_miter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);
	int iter = AMG_solve(Setup_data,b,X,NULL);
	AMG_free_data(Setup_data);
	return iter;
}*/

long double residual_norm(sparse_matrix* A,double* b,double* X,double* Weight){
	int n = glob_mesh.size;
	int N = A->size;
	double* V = set_vector_Ui_0_2D(N/n);
	double* R = clone_vector(b,N);
	linear_map(R,-1.,A,X);
	vector_pseudo_div(R,V,N);
	if (Weight!=NULL) vector_pseudo_mult(R,Weight,N);
	long double res = hp_euklid_norm(R,N);
	free(V);
	free(R);
	return res;
}

void update_vector(double* V,double* V_prev,int size,double r){
	double* DV = clone_vector(V,size);
	vector_add(DV,V_prev,-1.,size);
	double dist = euklid_norm(DV,size);
	vector_normalize(DV,size);
	if (dist>r){ 
		copy_vector_content(V_prev,V,0,0,size);
		vector_add(V,DV,r,size);
	}
	free(DV);
}

void relax_vector(double* V,double* V_prev,int size,double alpha){
	scalar_mult(alpha,V,size);
	vector_add(V,V_prev,1.-alpha,size);
}

void get_CH_step_equation(sparse_matrix** A,double** b,double* Y,double* Y0,double* H,double* U,double* T){
	int n = glob_mesh.size;
	double* C = &(Y[0]);
	
	*A = get_CH_linear_part(H,Y,T);											
	sparse_matrix* A_nonlin = get_CH_nonlinear_part(H,Y,T);						
	sparse_add(*A,A_nonlin,-1.);	
	*b = Jacobian_self_c(Y,T);
	linear_map(*b,-1.,A_nonlin,Y),
	linear_map(*b,1.,Phase_ID_c,Y0);
	linear_map(*b,-vis_c/dt,Phase_ID_cross,Y0);													// viscous part of chemical potential
		
	double* El_CH2 = zero_vector(2*n);															// elastic part of chemical potential
	double* El_CH = elastic_potential(U,H,C,T,PARTIAL0,PARTIAL1);
	
	double* Strain_energy_self = elastic_potential_self(U,H,C,T,PARTIAL0,PARTIAL1,PARTIAL0);	// self elastic part of chemical potential
	vector_add(El_CH,Strain_energy_self,1.,n);
	
	double* Heat_CH = free_energy_heat_dc(C,T);													// heat capacity part
	vector_add(El_CH,Heat_CH,1.,n);	
	
	double* Dam_CH = free_energy_damage_interface_dc(H,C);										// damage part
	vector_add(El_CH,Dam_CH,1.,n);	
	
	copy_vector_content(El_CH,El_CH2,0,n,n);
	vector_add(*b,El_CH2,1.,2*n);
		
	free_sparse(A_nonlin);
	free(Strain_energy_self);
	free(Heat_CH);
	free(Dam_CH);
	free(El_CH2);
	free(El_CH);
}

sparse_matrix* spread_damage_matrix(double* H){
	const double tol = 1e-1;
	int i,j,k,min;
	double h;
	int n = glob_mesh.size;
	sparse_matrix* Res = sparse_zero(n);
	for (i=0;i<n;i++){
		if (H[i]<1.-tol) insert_sparse(Res,1.,i,i);
		else{
			min = -1;
			h = 1.;
			for (j=0;j<glob_mesh.Sizes[i];j++){
				k = glob_mesh.Connections[i][j];
				if (H[k]<h && k!=i){
					min = k;
					h = H[k];
				}
			}
			if (min>=0 && h<1.-tol) insert_sparse(Res,1.,i,k); else insert_sparse(Res,1.,i,i);
		}
	}
	return Res;
}

void get_El_step_equation(sparse_matrix** A,double** b,double* C,double* H,double* U,double* T,bound_cond* BC_list_U){
	int n = glob_mesh.size;
	double* Bound_f = get_bound_force(BC_list_U);
	*A = elastic_potential_Hesse(U,H,C,T,PARTIAL0,PARTIAL0);	
	sparse_add(*A,Regularization,1.);
	*b = sparse_mult(Bound_Neumann,Bound_f);									// set force at Neumann boundary
	double* TE = Self_strain_div(U,H,C,T);										// include self stress		
	vector_add(*b,TE,1.,dimension*n);
	Set_Robin_BC(*A,*b,BC_list_U);									// set Robin conditions
	Set_dirichlet_BC(*A,*b,BC_list_U);								// set Dirichlet conditions
	free(TE);
	free(Bound_f);
}

void get_full_damage_system(sparse_matrix** A,double** b,double* C,double* H,double* H_prev,double* U,
 double* T,bound_cond* BC_list_U){
	int n = glob_mesh.size;
	double* V = set_vector_Ui_0_2D(1);
	sparse_matrix* K0 = elastic_potential_Hesse(U,H,C,T,PARTIAL0,PARTIAL0);
	sparse_matrix* ddXi = damage_self_hesse(H,C,n);

	double* dXi = damage_self_gradient(H,C,n);
	double* Alpha = damage_rate_independent(C);
	double* Kappa = damage_interface_tensor(H,C,PARTIAL0);
	double* Bound_f = get_bound_force(BC_list_U);
	double* dH = clone_vector(H,n);
	vector_add(dH,H_prev,-1.,n);
		
	sparse_matrix* Duxy_z = elastic_potential_du(U,H,C,T,PARTIAL1,PARTIAL0);
	sparse_matrix* Dz_uxy = get_transpose(Duxy_z,dimension*n);
	
	
	double* K = sparse3D_times_rank2_times_scalar_2D_right(Strain_grad,Kappa,H);
	sparse_matrix* Int = sparse3D_times_rank2_2D_right(Strain_grad,Kappa);	
		
	double* Z = generate_vector(n,beta/dt);
	double* E2 = elastic_potential(U,H,C,T,PARTIAL2,PARTIAL0);
	vector_pseudo_div(E2,V,n);
	vector_add(Z,E2,1.,n);
	sparse_matrix* Dzz = sparse3D_vB(T_Source,Z,n);
	sparse_add(Dzz,ddXi,1.);
	sparse_add(Dzz,Int,1.);
	free(E2);
	free(Z);

	*A = sparse_zero((dimension+1)*n);
	// elastic part
	add_subsparse_to_sparse(*A,K0,0,0);
	add_subsparse_to_sparse(*A,Regularization,0,0);
	add_subsparse_to_sparse(*A,Dz_uxy,0,dimension*n);
	
	// damage part
	add_subsparse_to_sparse(*A,Dzz,dimension*n,dimension*n);
	add_subsparse_to_sparse(*A,Duxy_z,dimension*n,0);
	
	*b = zero_vector((dimension+1)*n);
	// elastic part
	linear_map(&((*b)[0]),1.,Bound_Neumann,Bound_f);	
	linear_map(&((*b)[0]),1.,Dz_uxy,H);
	
	//damage part
	linear_map(&((*b)[dimension*n]),1.,Dissipation,Alpha);
	linear_map(&((*b)[dimension*n]),-beta/dt,Dissipation,dH);
	linear_map(&((*b)[dimension*n]),1.,Dzz,H);
	linear_map(&((*b)[dimension*n]),0.5,Duxy_z,U);
	vector_add(&((*b)[dimension*n]),dXi,-1.,n);
	
	// BC
	int i;
	for (i=0;i<BC_list_U->size;i++) if (BC_list_U->Cond[i]==DIRICHLET){
		(*b)[i] = BC_list_U->Val[i];
		reset_row(*A,i);
		insert_sparse(*A,1.,i,i);
	} 
	
	// include also other BC ! 
	
	// clean 
	free_sparse(K0);
	free_sparse(ddXi);
	free_sparse(Dz_uxy);
	free_sparse(Duxy_z);
	free_sparse(Dzz);
	free_sparse(Int);
	free(V);
	free(dXi);
	free(Alpha);
	free(Kappa);
	free(Bound_f);
	free(dH);
}

sparse_matrix* gradient_flux_add_constraints(sparse_matrix* A,sparse_matrix* C,sparse_matrix* CT,int* ActiveSet,int active_size){
	int i,j,k;
	int n = glob_mesh.size;
	int N = A->size;
	sparse_matrix* Res = clone(A);
	enlarge_matrix(Res,N,N+active_size,0,0);
	
	for (k=0;k<active_size;k++){
		i = ActiveSet[k];
		for (j=0;j<CT->Len[i];j++) insert_sparse(Res,CT->Values[i][j],n+CT->Indices[i][j],N+k);
		for (j=0;j<C->Len[i];j++) insert_sparse(Res,C->Values[i][j],N+k,C->Indices[i][j]);
	}
	
	return Res;
}

/*int gradient_flux_constraint_solve(sparse_matrix* A,double* Sol,double* b,sparse_matrix* B,double* L,double* U){
	const double tol = 1e-10;
	
	int i,j,k,fulfilled;
	sparse_matrix* M;

	int N= A->size;
	int m = B->size;	
	int n = glob_mesh.size;
	
	if (n!=m){
		printf("constraint matrix must be square matrix -> abort\n");
		exit(0);
	}
	
	int* Active = zero_int_list(m);
	double* Signs = zero_vector(m);
	double* Constraint = zero_vector(m);
	double* Z = clone_vector(b,N);
	double* X = zero_vector(N+m);
	copy_vector_content(Sol,X,0,0,N);
	sparse_matrix* BT = get_transpose(B,n);
	set_var_number2D(1);
	sparse_matrix3D* GR2 = set_matrix_Bijk_011(0,0,0,&insert_B_aaWWW);
	
	int iter = 0;
	int active_size = 0;
	do{
		int* ActiveSet = (int*)malloc(active_size*sizeof(int));
		double* ActiveSigns = (double*)malloc(active_size*sizeof(double));
		Z = realloc(Z,(N+active_size)*sizeof(double));
		k = 0;
		for (i=0;i<m;i++) if (Active[i]){			
			ActiveSet[k] = i;
			ActiveSigns[k] = Signs[i];
			Z[N+k] = Constraint[i];
			k++;
		}
		if (k!=active_size) printf("\nsomenthing's wrong in active_set_solver\n");
		M = gradient_flux_add_constraints(A,B,BT,ActiveSet,active_size);
		
		multifrontal_solver_RN(M,X,Z);
		fulfilled = 1;
		
		// test sign of KKT multipliers -> delete constraints if not feasible	
		double* K = zero_vector(m);
		for (k=0;k<active_size;k++){
			i = ActiveSet[k];
			K[i] = X[N+k];
		}
		double* BTK = sparse_mult(BT,K);
		double* dF = zero_vector(n);
		double* Cond_b = sparse3D_bilinear(GR2,BTK,&(X[n]));
		multifrontal_solver_RN(ID_damage,dF,Cond_b);
		free(Cond_b);
		free(BTK);
		free(K);
		for (k=0;k<active_size;k++){
			i = ActiveSet[k];
			if (dF[i]<0){
				Active[ActiveSet[k]] = 0;
				active_size--;
				fulfilled = 0;
			}
		}
		
		// test constraints -> if not fulfilled add constraints
		double* Y = sparse_mult(B,X);
		for (i=0;i<m;i++){
			if (Y[i]<L[i]-tol || Y[i]>U[i]+tol){		
				if (!Active[i]) active_size++;			
				Active[i] = 1;
				if (Y[i]<L[i]-tol){
					Signs[i] = 1.;
					Constraint[i] = L[i];
				}
				if (Y[i]>U[i]+tol){
					Signs[i] = -1.;
					Constraint[i] = U[i];
				}
				fulfilled = 0;
			}
		}
		
		free(Y);
		free(dF);
		free(ActiveSet);
		free(ActiveSigns);	
		free_sparse(M);
		iter++;
		
	}while(!fulfilled && iter<R_Newton_max_iter);
	
	copy_vector_content(X,Sol,0,0,N);
	if (iter>=R_Newton_max_iter) printf("Warning: no convergence in %d iterations\n",iter);
	
	free_sparse3D(GR2);
	free_sparse(BT);
	free(Active);
	free(Signs);
	free(Constraint);
	free(X);
	free(Z);
	return iter;
}*/

/*int gradient_flux_constraint_solve(sparse_matrix* A,double* Sol,double* Sol0,double* b,double* L,double* U,
 int* C_index,int c_size){
	const double tol = 1e-10;
	
	int i,j,k,fulfilled;
	sparse_matrix* M;

	int N= A->size;
	int m = c_size;;	
	int n = glob_mesh.size;
	
	if (n!=m){
		printf("constraint matrix must be square matrix -> abort\n");
		exit(0);
	}
	
	int* Active = zero_int_list(m);
	double* Signs = zero_vector(m);
	double* Constraint = zero_vector(m);
	double* Z = zero_vector(N+m);
	copy_vector_content(b,Z,0,0,N);
	double* X = zero_vector(N+m);
	copy_vector_content(Sol,X,0,0,N);
	double* X0 = Sol0;
	
	sparse_matrix* B = sparse_identity(m);
	sparse_matrix* BT = get_transpose(B,n);
	
	int iter = 0;
	int active_size = 0;
	do{
		multifrontal_solver_RN(A,X,b);
		double* Y = sparse_mult(B,X);
		for (j=0;j<m;j++){
			i = C_index[j];
			if (X[i]-X0[i]<0) Signs[j] = -1.; else Signs[j] = 1.;
			if (Y[j]<L[j]-tol || Y[j]>U[j]+tol){
				Active[j] = 1;
				active_size++;
				if (Y[j]<L[j]-tol) Constraint[j] = L[j];
				if (Y[j]>U[j]-tol) Constraint[j] = U[j];
			}
		}
		
		int* ActiveSet = (int*)malloc(active_size*sizeof(int));
		k = 0;
		for (j=0;j<m;j++) if (Active[j]){			
			ActiveSet[k] = j;
			ActiveSigns[k] = Signs[j];
			Z[N+k] = Constraint[j];
			k++;
		}
		if (active_size==0){
			fulfilled = 1;
			goto FINALIZE;
		}
		
		M = gradient_flux_add_constraints(A,B,BT,ActiveSet,active_size);
		multifrontal_solver_RN(M,X,Z);		
		
		
		
		
		
		
		
		
		
	FINALIZE: 
		free(Y);
		free(ActiveSet);
		free_sparse(M);
		iter++;
		
	}while(!fulfilled && iter<R_Newton_max_iter);
	
	copy_vector_content(X,Sol,0,0,N);
	if (iter>=R_Newton_max_iter) printf("Warning: no convergence in %d iterations\n",iter);
	
	free_sparse(B);
	free_sparse(BT);
	free(Active);
	free(Signs);
	free(Constraint);
	free(X);
	free(Z);
	return iter;
}*/

void El_step_line_search(sparse_matrix** A,double** b,double* C,double* H,double* U,double* T,bound_cond* BC_list_U,double* r,double* eq){
	int status;
	double lin,nonlin,ratio;
	
	int n = glob_mesh.size;
	*r = Alt_trust_max;
	double eq_prev = *eq;
	double* U_prev = clone_vector(U,dimension*n);
	do{
		status = multifrontal_solver(*A,U,*b,"alternate min: elastic step",&UMFPACK_time_elastic);		
		update_vector(U,U_prev,dimension*n,*r);
		lin = (double)residual_norm(*A,*b,U,NULL)-eq_prev;
		
		free_sparse(*A);
		free(*b);		
		get_El_step_equation(A,b,C,H,U,T,BC_list_U);
		*eq = (double)residual_norm(*A,*b,U,NULL);
		nonlin =  *eq-eq_prev;
		ratio = nonlin/fabs(lin);
		if (ratio>-Alt_trust_thres_min) *r /= 10.;
		if (ratio>-Alt_trust_thres_max/10.) copy_vector_content(U_prev,U,0,0,dimension*n);
	}while(ratio>-Alt_trust_thres_max/10.);
	
	/*double curv = 0;
	double* Dir = generate_vector((*A)->size,1.);
	vector_normalize(Dir,(*A)->size);
	get_neg_curvature_direction(*A,&curv,Dir,10000,1e-10,1e-10);
	free(Dir);*/
	//printf("line search U: new residuum=%e, new radius=%e, reduction ratio=%e\n",*eq,*r,ratio);
	free(U_prev);
}

int push_to_stack(double* X){
	
	const int mem = 5;
	const double tol = 1e-5;
	
	static int counter = 0;
	static double* Stack = NULL;
	
	int n = glob_mesh.size;
	
	if (X==NULL){
		if (Stack!=NULL) free(Stack);
		Stack = zero_vector(mem*n);
		counter = 0;
		return 0;
	}
	else{		
		memcpy(&(Stack[n]),Stack,(mem-1)*n*sizeof(double));
		memcpy(Stack,X,n*sizeof(double));
		counter++;
		
		if (counter>mem){
			int i;
			double d;
			
			int imin = 2;
			double d0 = vec_dist(Stack,&(Stack[n]),n);
			double dmin = vec_dist(Stack,&(Stack[2*n]),n);
			
			for (i=3;i<mem;i++){
				d = vec_dist(Stack,&(Stack[i*n]),n);
				if (d<dmin){
					dmin = d;
					imin = i;
				}
			}
			printf("lc indicator: %e\n",dmin/d0);
			if (dmin/d0<tol) return imin; else return 0;			
		}
		else return 0;
	}
}

int direct_alternate_minimization(double* X){	
	
	const double q = 1.0;
	
	static int debug = 0;
	
	#define SAVE { \
		if (debug){ \
			double* W = zero_vector(degrees_of_freedom*n); \
			double* dU = clone_vector(U,2*n); \
			vector_add(dU,U_prev,-1.,2*n); \
			copy_vector_content(dU,W,0,0,dimension*n); \
			double* dH = clone_vector(H,n); \
			vector_add(dH,H_prev,-1.,n); \
			copy_vector_content(dH,W,0,dimension*n,n); \
			free(dU); \
			free(dH); \
			double* dY = clone_vector(Y,2*n); \
			vector_add(dY,Y_prev,-1.,2*n); \
			copy_vector_content(dY,W,0,(dimension+1)*n,n); \
			free(dY); \
			double* dT = clone_vector(T,n); \
			vector_add(dT,T_prev,-1.,n); \
			copy_vector_content(dT,W,0,(dimension+3)*n,n); \
			free(dT); \			
			save(W,NULL,global_size,glob_time); \
			free(W); \		
		} \
	}
	
	#ifdef CONV_CHECK
		int k;
		double** R = (double**)malloc(Alt_max_iter*sizeof(double*));
		for (k=0;k<Alt_max_iter;k++) R[k] = NULL;
	#endif
	
	//const int many_iter = 100;
	//const double eps_tol = 1e-6;
	
	
	int i,j,iter,iter2,last_iter,status,constrl,constru;
	double res,res1,res2,res3,res4,g,bi,bl,bu,c,d,th;
	double* Alpha;
	double* C_prev;
	double* b_step1;
	//double* b_step2;
	double* b_step3;
	double* b_step4;
	double* dG;
	double* El_CH;
	double* El_CH2;
	double* Heat_CH;
	double* Strain_energy_self;
	//double* Damage_right_side;
	double* Temp;
	
	long double red_U,red_Y,red_T,eqT,eqU,eqY,temp;
	
	int continuous = 1;
	
	double trust_radius_Y = Alt_trust_max;
	double trust_radius_U = Alt_trust_max;
	double trust_radius_T = Alt_trust_max;
	
	/*long double eqY = 1e20;
	long double eqU = 1e20;
	long double eqT = 1e20;*/
	
	//sparse_matrix* C_matrix = NULL;
	sparse_matrix* A_step1 = NULL;
	//sparse_matrix* A_step2 = NULL;
	sparse_matrix* A_step3 = NULL;
	sparse_matrix* A_step4 = NULL;
	sparse_matrix* A_nonlin = NULL;
	sparse_matrix* Damage_constr = NULL;
	
	int n = glob_mesh.size;	
	int n_prev = Glob_mesh_sizes[glob_time_index-1];
	double* V = set_vector_Ui_0_2D(1);
	
	double* U0 = &(X[0]);
	double* H0 = &(X[dimension*n]);
	double* C0 = &(X[(dimension+1)*n]);
	double* Y0 = &(X[(dimension+1)*n]);
	double* T0 = &(X[(dimension+3)*n]);

#ifndef CONSTANT_PHASE
	double* C_Lower = generate_vector(n,minimum_c);							// constraints c in [min,max]
	double* C_Upper = generate_vector(n,maximum_c);
#endif
	force_constraints(C0,minimum_c,maximum_c);								// force constraints if violated after interpolation
	
	// working arrays
	double* H = zero_vector(n);
	double* H_prev = zero_vector(n);
	double* C = zero_vector(n);	
	double* T_prev = zero_vector(n);	
	double* U = zero_vector(dimension*n);
	double* U_prev = zero_vector(dimension*n);
	double* Kernel_step2 = generate_vector(n,1.);
	double* T = clone_vector(T0,n);
	double* Y = zero_vector(2*n);
	double* Y_prev = clone_vector(Y,2*n);
	copy_vector_content(X,U,0,0,dimension*n);
	copy_vector_content(X,H,dimension*n,0,n);
	copy_vector_content(X,C,(dimension+1)*n,0,n);
	copy_vector_content(X,Y,(dimension+1)*n,0,2*n);
	
	// init damage step
	Glob_Pointer_H0 = H0;
	
	double* Hupper = NULL;
	double* Hlower = NULL;
	ass_info* Damage_solver_info = NULL;
	if (strain_decomposition==EIGENSYSTEM) set_total_strain_eigensystem(U,C,T,&Glob_eigensystem0);
	else if (strain_decomposition==TRACE) Glob_Pointer_Trace0 = filtered_total_strain_trace(U0,H0,C0,T0);	
	
	// init elastic step
	bound_cond* BC_list_U = get_U_bound_conditions(glob_time);
	sparse_matrix* Pattern = elastic_potential_Hesse(U,H,C,T,PARTIAL0,PARTIAL0);
	sparse3D_times_rank4_2D_right_get_map(Strain_grad,Pattern,&El_Hesse_Map);
	free_sparse(Pattern);
	get_El_step_equation(&A_step1,&b_step1,C,H,U,T,BC_list_U);
	long double eqU_init = residual_norm(A_step1,b_step1,U,NULL);
	eqU_init = (eqU_init==0) ? 1. : eqU_init;
	eqU = eqU_init;
	free_sparse(A_step1);
	free(b_step1);
	
	// init chemical step
#ifndef CONSTANT_PHASE
	get_CH_step_equation(&A_step3,&b_step3,Y,Y0,H,U,T);
	long double eqY_init = residual_norm(A_step3,b_step3,Y,NULL);
	eqY_init = (eqY_init==0) ? 1. : eqY_init;
	eqY = eqY_init;
	
	ass_info* CH_solver_info = NULL;
	double* D = generate_vector(n,1.);
	sparse_matrix* CH_constraints = sparse_diagonal(D,n);
	free(D);
	
	//reflect(&(Y[0]),-1.,1.);												// only necessary if potential has singularities at bounds

	free_sparse(A_step3);
	free(b_step3);	
#endif
	
	// init temperature step
	bound_cond* BC_list_T = NULL;											// involves current T
#ifndef NO_TEMP
	A_step4 = get_Tdyn_left_side_matrix(H,H0,C,C0,U,U0,T);
	b_step4 = get_Tdyn_right_side_vector(H,H0,C,C0,&(Y[n]),U,U0,T,T0);			
	BC_list_T = get_T_bound_conditions(T,glob_time);
	Tdyn_apply_BC(A_step4,b_step4,BC_list_T);
	long double eqT_init = residual_norm(A_step4,b_step4,T,NULL);
	eqT_init = (eqT_init==0) ? 1. : eqT_init;
	eqT = eqT_init;
	free_bound_cond(&BC_list_T);
	free_sparse(A_step4);
	free(b_step4);
#endif	
		
	// constraint solver settings
	RN_set_local_solver(&multifrontal_solver_RN);								// set Multifrontal solver
	R_Newton_Condition = zero_int_list(n);
		
	int* Indices = get_all_indices(n);											// set constraints
	double* Lower = zero_vector(n);
	double* Upper = interpolated(All_H[glob_time_index-1],1,n_prev,NULL,NULL);	// necessary or just H0 ?
	force_constraints(Upper,0,1.);												// force constraints after interpolation
	
	switch(constraint_solver){
		case REFLECTIVE_NEWTON:
			set_RN_constraints(Indices,Lower,Upper,n);
			set_RN_functional(&damage_functional);
			RN_set_gradient(&damage_gradient);
			RN_set_Hesse_function(&damage_Hesse,n);
			detect_lower_equal_upper(Lower,Upper,R_Newton_Condition,NULL,n,R_Newton_Dir_tol);	// if upper==lower then use Dirichlet		
			break;
		case ACTIVE_SET:
			Damage_constr = sparse_zero((dimension+1)*n);
			Hupper = generate_vector((dimension+1)*n,1.);
			Hlower = generate_vector((dimension+1)*n,-1.);
			for (j=dimension*n;j<(dimension+1)*n;j++){
				insert_sparse(Damage_constr,1.,j,j);
				Hlower[j] = 0;
				Hupper[j] = Upper[j-dimension*n];
			}			
			break;
	}	

	iter = 0;
	iter2 = 0;
	last_iter = 0;
	//push_to_stack(NULL);
	
	do{													
		//first step: Cahn Hilliard dynamics 																																																																						
		copy_vector_content(Y,Y_prev,0,0,2*n);								// CH kinetics
	#ifndef CONSTANT_PHASE
		get_CH_step_equation(&A_step3,&b_step3,Y,Y0,H,U,T);
		
		//double* V = set_vector_Ui_0_2D(1);
		//double d0 = scalar(V,Y,n);
		
		//status = multifrontal_solver(A_step3,Y,b_step3,"alternate min: chemical step",&UMFPACK_time_chemistry);		
		status = active_set_solver(A_step3,Y,b_step3,CH_constraints,C_Lower,C_Upper,&multifrontal_solver_AS
		 ,R_Newton_max_iter,NEG_DEFINITE,ASS_FLUX,&CH_solver_info);		
		 
		relax_vector(Y,Y_prev,2*n,1.0);
		 
 		/*double d = scalar(V,Y,n);
 		printf("iter: %d\t c diff: %e\n",status,(d-d0)/d0);
 		free(V);*/

		
		temp = residual_norm(A_step3,b_step3,Y,NULL)-eqY;
		Temp = clone_vector(Y,2*n);		
		if (Alt_trust_region==YES) update_vector(Y,Y_prev,2*n,trust_radius_Y);	
		long double eq_diff_Y = residual_norm(A_step3,b_step3,Y,NULL)-eqY;
		if (eq_diff_Y>0){
			copy_vector_to(Temp,Y,2*n);	
			eq_diff_Y = temp;
		}
		free_sparse(A_step3);
		free(b_step3);
		free(Temp);
	#endif
		copy_vector_content(Y,C,0,0,n);
		Glob_Pointer_C = C;
		
		// fourth step: temperature evolution
		copy_vector_to(T,T_prev,n);
	#ifndef NO_TEMP
		double* T00 = clone_vector(T0,n);
		//temp_exp_mult(T00,H,H0,C,C0,U,U0,T);
		A_step4 = get_Tdyn_left_side_matrix(H,H0,C,C0,U,U0,T);
		b_step4 = get_Tdyn_right_side_vector(H,H0,C,C0,&(Y[n]),U,U0,T,T00);			
		BC_list_T = get_T_bound_conditions(T,glob_time);
		Tdyn_apply_BC(A_step4,b_step4,BC_list_T);
		status = multifrontal_solver(A_step4,T,b_step4,"alternate min: heat step",&UMFPACK_time_temperature);			
		
		relax_vector(T,T_prev,n,1.0);
				
		temp = residual_norm(A_step4,b_step4,T,NULL)-eqT;
		Temp = clone_vector(T,n);			
		if (Alt_trust_region==YES) update_vector(T,T_prev,n,trust_radius_T);	
		for (i=0;i<n;i++) if (T[i]<0){
			printf("Warning: negative temperature at node index %d (x=%f,y=%f) -> abort\n",i,glob_mesh.Points[i].x,glob_mesh.Points[i].y);
			print_vector(T,n);
			exit(0);
		}
		
		long double eq_diff_T = residual_norm(A_step4,b_step4,T,NULL)-eqT;
		if (eq_diff_T>0){
			copy_vector_to(Temp,T,n);	
			eq_diff_T = temp;
		}

		free_sparse(A_step4);
		free(b_step4);
		free(Temp);
		free(T00);
	#endif
		
		long double eq_diff_U;
		if (constraint_solver==ACTIVE_SET){
			copy_vector_to(U,U_prev,dimension*n);	
			copy_vector_to(H,H_prev,n);	
			double* Z = zero_vector((dimension+1)*n);
			copy_vector_content(U,Z,0,0,dimension*n);
			copy_vector_content(H,Z,0,dimension*n,n);
			double* Z_prev = clone_vector(Z,(dimension+1)*n);
			get_full_damage_system(&A_step1,&b_step1,C,H,H0,U,T,BC_list_U);
	
			status = active_set_solver(A_step1,Z,b_step1,Damage_constr,Hlower,Hupper,&multifrontal_solver_RN
			,R_Newton_max_iter,POS_DEFINITE,ASS_DEFAULT,&Damage_solver_info);
			printf("acive set solver (U,H): % iterations\n",status);
		
			scalar_mult(q,Z,(dimension+1)*n);
			vector_add(Z,Z_prev,(1.-q),(dimension+1)*n);
		
			copy_vector_content(Z,U,0,0,dimension*n);
			copy_vector_content(Z,H,dimension*n,0,n);
			free(Z);
			free(Z_prev);			
			
			eq_diff_U = 1.;
			free_sparse(A_step1);
			free(b_step1);
		}
		if (constraint_solver==REFLECTIVE_NEWTON){
			// second step: Elastic equation							
				
			copy_vector_to(U,U_prev,dimension*n);								
			get_El_step_equation(&A_step1,&b_step1,C,H,U,T,BC_list_U);
			status = multifrontal_solver(A_step1,U,b_step1,"alternate min: elastic step",&UMFPACK_time_elastic);			
			
			relax_vector(U,U_prev,dimension*n,1.0);
		
			temp = residual_norm(A_step1,b_step1,U,NULL)-eqU;
			Temp = clone_vector(U,dimension*n);		
			if (Alt_trust_region==YES) update_vector(U,U_prev,dimension*n,trust_radius_U);	
				
			eq_diff_U = residual_norm(A_step1,b_step1,U,NULL)-eqU;
			if (eq_diff_U>0){
				copy_vector_to(Temp,U,dimension*n);	
				eq_diff_U = temp;
			}
			free_sparse(A_step1);
			free(b_step1);
			free(Temp);		
		
			// third step: damage evolution		
			Strain_energy = elastic_potential(U,H,C,T,FACTOR,PARTIAL0);
			Damage_Hesse_linear = get_damage_Hesse_linear(C);
			scalar_mult(2.,Strain_energy,n);		
			Strain_energy_self = elastic_potential_self(U,H,C,T,FACTOR,PARTIAL0,PARTIAL0);
			vector_add(Strain_energy,Strain_energy_self,2.,n);	
			free(Strain_energy_self);			
		
			copy_vector_to(H,H_prev,n);											// constrained solver	
			RN_set_trust_region(R_Newton_Delta);										
			RN_set_kernel(Kernel_step2,n);
			iter2 += reflective_newton(H);	
			
			free_sparse(Damage_Hesse_linear);
			free(Strain_energy);
		}
		
		if (Alt_trust_region==YES) update_vector(H,H_prev,n,trust_radius_U);
	
		//SAVE
		
		//copy_vector_to(H_prev,H,n);												// Attention: damage dynamics disabled
		
		if (Alt_trust_region==YES){
			
			// compute reduction ratio
		#ifndef CONSTANT_PHASE
			get_CH_step_equation(&A_step3,&b_step3,Y,Y0,H,U,T);			
			long double eqY0 = eqY;
			//A_step3->size = n;														// use only firs component of A
			eqY = residual_norm(A_step3,b_step3,Y,NULL);
			//A_step3->size = 2*n;
			free_sparse(A_step3);
			free(b_step3);
			red_Y = (eqY-eqY0)/eq_diff_Y;		
		#endif
			
		#ifndef NO_TEMP
			A_step4 = get_Tdyn_left_side_matrix(H,H0,C,C0,U,U0,T);
			b_step4 = get_Tdyn_right_side_vector(H,H0,C,C0,&(Y[n]),U,U0,T,T0);			
			Tdyn_apply_BC(A_step4,b_step4,BC_list_T);			
			long double eqT0 = eqT;
			eqT = residual_norm(A_step4,b_step4,T,NULL);
			free_sparse(A_step4);
			free(b_step4);
			red_T = (eqT-eqT0)/eq_diff_T;			
			free_bound_cond(&BC_list_T);
		#endif
		
			get_El_step_equation(&A_step1,&b_step1,C,H,U,T,BC_list_U);					
			long double eqU0 = eqU;
			eqU = residual_norm(A_step1,b_step1,U,NULL);
			red_U = (eqU-eqU0)/eq_diff_U;				
			free_sparse(A_step1);
			free(b_step1);		
			
			// modify trust region			
			double factor;
			if (trust_control_radius==NO_EXPAND) factor = 1.; else factor = Alt_trust_decrease;			
		
		#ifndef CONSTANT_PHASE
			if (eq_diff_Y==0) red_Y = eqY-eqY0;
			//else if (red_Y>-Alt_trust_thres_min){			
			if (red_Y<Alt_trust_min){
				trust_radius_Y *= Alt_trust_decrease;
				if (trust_radius_Y<Alt_trust_min) trust_radius_Y = Alt_trust_min;
			}
			//else if (red_Y<-Alt_trust_thres_max){
			if (red_Y>Alt_trust_max && eq_diff_Y!=0){
				trust_radius_Y /= factor;
				if (trust_radius_Y>Alt_trust_max) trust_radius_Y = Alt_trust_max;
			}
		#endif
			
		#ifndef NO_TEMP
			if (eq_diff_T==0) red_T = eqT-eqT0;
			//else if (red_T>-Alt_trust_thres_min)){
			if (red_T<Alt_trust_min){
				trust_radius_T *= Alt_trust_decrease;
				if (trust_radius_T<Alt_trust_min) trust_radius_T = Alt_trust_min;
			}
			//if (red_T<-Alt_trust_thres_max){
			if (red_T>Alt_trust_max && eq_diff_T!=0){
				trust_radius_T /= factor;
				if (trust_radius_T>Alt_trust_max) trust_radius_T = Alt_trust_max;
			}
		#endif
			if (eq_diff_U==0) red_U = eqU-eqU0;					
			//else if (red_U>-Alt_trust_thres_min){
			if (eq_diff_U<0 && red_U<Alt_trust_min){
				trust_radius_U *= Alt_trust_decrease;
				if (trust_radius_U<Alt_trust_min) trust_radius_U = Alt_trust_min;
			}
			//if (red_U<-Alt_trust_thres_max){
			if (red_U>Alt_trust_max && eq_diff_U!=0){
				trust_radius_U /= factor;
				if (trust_radius_U>Alt_trust_max) trust_radius_U = Alt_trust_max;
				
			}				
			
			// reject solution if set
			int Tcond,Ycond;
			continuous = 1;
			switch(trust_control_rejection){
				case NO_REJECTION:break;
				case PARTIAL_REJECTION:
		#ifndef CONSTANT_PHASE
					if (eqY-eqY0>0){
						copy_vector_to(Y_prev,Y,2*n);
						//printf("rejetc Y (old=%Le, new=%Le)\n",eqY0,eqY);
						eqY = eqY0;				 
					}
		#endif
		#ifndef NO_TEMP
					if (eqT-eqT0>0){
						copy_vector_to(T_prev,T,n);
						eqT = eqT0;				 
					}
		#endif
					if (eqU-eqU0>0){
						//printf("rejetc U (old=%Le, new=%Le)\n",eqU0,eqU);
						copy_vector_to(U_prev,U,dimension*n);
						copy_vector_to(H_prev,H,n);
						eqU = eqU0;			
						continuous = (trust_radius_U==Alt_trust_min) ? 0 : 1;	 
					}
					break;
				case TOTAL_REJECTION:
					Tcond = 0;
					Ycond = 0;
		#ifndef NO_TEMP
					Tcond = (eqT-eqT0);
		#endif
		#ifndef CONSTANT_PHASE
					Ycond = (eqY-eqY0);
		#endif
					if (eqU-eqU0>0 || Ycond || Tcond){
						//printf("rejetc all\n");						
						copy_vector_to(U_prev,U,dimension*n);
						copy_vector_to(H_prev,H,n);						
						eqU = eqU0;		
						continuous = (trust_radius_U==Alt_trust_min) ? 0 : 1;	 
		#ifndef NO_TEMP						
						copy_vector_to(T_prev,T,n);
						eqT = eqT0;	
		#endif
		#ifndef CONSTANT_PHASE						
						copy_vector_to(Y_prev,Y,2*n);
						eqY = eqY0;	
		#endif
					}
					break;				
			}
			
			//printf("residuals:  eqU: %Le, eqY: %Le, rad U: %e, rad Y: %e, red U %Le, red Y: %Le\n",eqU,eqY,trust_radius_U,trust_radius_Y,red_U,red_Y);
			printf("residuals:  eqU: %Le, rad U: %e,red U %Le, diff U %Le\n",eqU/eqU_init,trust_radius_U,red_U,(long double)vec_dist(U,U_prev,dimension*n));
			
			// residuum		
			res = eqU/eqU_init;
		#ifndef CONSTANT_PHASE
			res = (eqY/eqY_init>res) ? eqY/eqY_init : res;
		#endif		
		#ifndef NO_TEMP
			res = (eqT/eqT_init>res) ? eqT/eqT_init : res;
		#endif											
		}
		else{	
			double old_res = res;	
			if (BC_list_T!=NULL) free_bound_cond(&BC_list_T);
			
			res1 = vec_dist(U,U_prev,dimension*n);
			res2 = vec_dist(H,H_prev,n);
			res = res1*res1+res2*res2;
		#ifndef CONSTANT_PHASE
			res3 = vec_dist(Y,Y_prev,2*n);			
			res += res3*res3;
		#endif
		#ifndef NO_TEMP
			res4 = vec_dist(U,U_prev,dimension*n);
			res += res4*res4;
		#endif
			res = sqrt(res/4.);	
			if (fabs(res-old_res)/res<1e-7){
				printf("periodic orbit encountered: escape iteration cycle");
				iter=Alt_max_iter;
			}
			printf("residual: %e\n",res);		
		}
		
	#ifdef CONV_CHECK
		R[iter] = clone_vector(H,n);
	#endif
						
		iter++;				
		//if (iter<many_iter) alt_eps = Alt_eps; else alt_eps = eps_tol;								
	}while(res>Alt_eps && iter<Alt_max_iter && continuous);
	if (Alt_info==YES){
		printf("\n\nAlternate Minimization: residuum %e iterations %d\n",res,iter);
		printf("Constraint solver average iterations: %f\n",(double)iter2/iter);
	}
	if (iter<Alt_max_iter || dt<=dt_min) AMG_add_stat(&alter_min_stat,iter,res);
	
	// copy to argument
	copy_vector_content(U,X,0,0,dimension*n);
	copy_vector_content(H,X,0,dimension*n,n);		
	copy_vector_content(C,X,0,(dimension+1)*n,n);
	copy_vector_content(Y,X,n,(dimension+2)*n,n);
	copy_vector_content(T,X,0,(dimension+3)*n,n);	
	
	Glob_Pointer_H0 = NULL;
	Glob_Pointer_C = NULL;
	
	// clean
	switch(constraint_solver){
		case REFLECTIVE_NEWTON:
			break;
		case ACTIVE_SET:
			if (Damage_constr!=NULL) free_sparse(Damage_constr);
			if (Hupper!=NULL) free(Hupper);
			if (Hlower!=NULL) free(Hlower);
			break;
	}
	free_ass_info(&Damage_solver_info);
	free(V);
	free(H);
	free(H_prev);
	free(U);
	free(U_prev);
	free(Y);
	free(Y_prev);
	free(C);
	free(T);
	free(T_prev);
	free(Upper);
	free(Lower);
	free(Indices);
	free(Kernel_step2);
	free(R_Newton_Condition);
	free_bound_cond(&BC_list_U);
	free_sparse_index_map(&El_Hesse_Map);
	Glob_Pointer_Trace0 = NULL;
	R_Newton_Condition = NULL;
	
#ifndef CONSTANT_PHASE
	free_ass_info(&CH_solver_info);
	free_sparse(CH_constraints);
	free(C_Lower);
	free(C_Upper);
#endif	

#ifdef CONV_CHECK
	char Name[512];
	double* Fin;
	sprintf(Name,"%s/%s/conv",Output_dir,Output_name);
	FILE* file = fopen(Name,"a");
	if (iter<Alt_max_iter){
		Fin = R[iter-1]; 
		d = euklid_norm(Fin,n);
		if (d==0) d = 1.;
	}
	else Fin = NULL;
	fprintf(file,"%e\t%d\t%e",dt,iter,res);
	if (Fin!=NULL) for (i=0;i<iter-1;i++) fprintf(file,"\t%e",log(vec_dist(R[i],Fin,n)/d)/log(10.));
	fprintf(file,"\n");
	
	/*if (iter>150){
		i = 0;
		while (i<iter){
			sprintf(Name,"%s/%s/U%d",Output_dir,Output_name,i/50);
			if (i==150) vector_add(R[i],R[50],-1.,n);			
			print_scalar_data(Name,R[i],n);
			i += 50;
		}
		exit(0);
	}*/
	
	for (i=0;i<Alt_max_iter;i++) if (R[i]!=NULL) free(R[i]);
	free(R);
	fclose(file);
#endif
	
	if (strain_decomposition==EIGENSYSTEM) free_eigen_field(&Glob_eigensystem0);
	else if (strain_decomposition==TRACE) free(Glob_Pointer_Trace0);
	
	if (status==FAILED) return FAILED; else return iter;
}

//energy estimation decreasing test, maybe there is a summation error
int energy_condition_fulfilled(double* X,double* Globals){
	int i,constrl,constru;
	double g,bi,bl,bu,c,d,th,s;
	
	int n_prev = Glob_mesh_sizes[glob_time_index-1];
	int n = glob_mesh.size;
	double* U = &(X[0]);
	double* H = &(X[dimension*n]);
	double* C = &(X[(dimension+1)*n]);
	double* Y = &(X[(dimension+2)*n]);											// has only one component
	double* T = &(X[(dimension+3)*n]);	
	double* U_prev = interpolated(All_U[glob_time_index-1],dimension,n_prev,NULL,NULL);
	double* H_prev = interpolated(All_H[glob_time_index-1],1,n_prev,NULL,NULL);
	double* C_prev = interpolated(All_C[glob_time_index-1],1,n_prev,NULL,NULL);
	double* T_prev = interpolated(All_T[glob_time_index-1],1,n_prev,NULL,NULL);
	
	// allocate memory for energy variables
	if (glob_time_index>=(int)time_steps){
		dissipation_damage = (double*)realloc(dissipation_damage,(glob_time_index+1)*sizeof(double));
		dissipation_chemical = (double*)realloc(dissipation_chemical,(glob_time_index+1)*sizeof(double));
		heat_bound_flux = (double*)realloc(heat_bound_flux,(glob_time_index+1)*sizeof(double));
		energy_constr_lower = (double*)realloc(energy_constr_lower,(glob_time_index+1)*sizeof(double));
		energy_constr_upper = (double*)realloc(energy_constr_upper,(glob_time_index+1)*sizeof(double));
		entropy_production = (double*)realloc(entropy_production,(glob_time_index+1)*sizeof(double));
	}
	
	// compute global energies
	c = Dissipation_chemical(Y,H,C_prev,C,T);
	d = Damage_dissipation(H_prev,H,C);
	g = Total_free_energy(U,H,C,T);
	bi = Boundary_damage(H_prev,H);
	bl = Boundary_elastic(U_prev,U,H,C,T_prev,T)+bi;
	bu = Boundary_elastic(U_prev,U,H_prev,C,T_prev,T)+bi;
	th = 0;
	s = Entropy_production_rate(U,U_prev,H,H_prev,C,C_prev,Y,T,T_prev)*dt;
	
	energy_constr_lower[glob_time_index] = energy_constr_lower[glob_time_index-1]+bl;
	energy_constr_upper[glob_time_index] = energy_constr_upper[glob_time_index-1]+bu;
	dissipation_damage[glob_time_index] = dissipation_damage[glob_time_index-1]+d;
	dissipation_chemical[glob_time_index] = dissipation_chemical[glob_time_index-1]+c;
	heat_bound_flux[glob_time_index] = heat_bound_flux[glob_time_index-1]+th;
	entropy_production[glob_time_index] = entropy_production[glob_time_index-1]+s;
	constrl = (dissipation_damage[glob_time_index]+g-free0>=energy_constr_lower[glob_time_index]);
	constru = (dissipation_damage[glob_time_index]+g-free0<=energy_constr_upper[glob_time_index]);
	
	if (Alt_info==YES && backtracking && (!constrl || !constru)){
		printf("\nenergy condition violated: step back to time %f\n\n",glob_time-dt);
	}
	
	// copy results to argument
	Globals[0] = Free_energy_elastic(U,H,C,T);
	Globals[1] = Free_energy_damage(H,C);
	Globals[2] = Free_energy_chemical(C,T);
	Globals[3] = Free_energy_heat(C,T);
	Globals[4] = dissipation_damage[glob_time_index];
	Globals[5] = dissipation_chemical[glob_time_index];
	Globals[6] = heat_bound_flux[glob_time_index];
	Globals[7] = energy_constr_lower[glob_time_index];
	Globals[8] = energy_constr_upper[glob_time_index];
	Globals[9] = Total_free_energy(U,H,C,T)-free0;
	Globals[10] = entropy_production[glob_time_index];
	Globals[11] = Total_entropy(U,H,C,T)-entropy0;
	Globals[12] = Inner_Energy(U,H,C,T)-inner0;
	Globals[13] = get_constraint_work(X);
	Globals[14] = mean_bound_displacement(U_prev,U).x;
	Globals[15] = Boundary_elastic_force(U_prev,U,H,C,T_prev,T).x;
	Globals[16] = extremal_crack_position_y(H);//damage_integral(H);
	Globals[17] = Thermal_energy(U,H,C,T);
	
	// clean
	free(C_prev);
	free(H_prev);
	free(U_prev);
	free(T_prev);
	
	if (!backtracking || (backtracking && constrl && constru)) return 1; else return 0;
}

void Load_mesh(int index,mesh2D* Mesh,element_collection* Triangles,int quiet){
	int i,len;
	char Name[512];
	char Dir[512];
	
	char** Parts = split(Mesh_name,".",&len);
	sprintf(Dir,"%s/%s/meshes",Output_dir,Output_name);
	sprintf(Name,"%s.%d",Parts[0],index);
	
	read_mesh_2D(Dir,Name,Mesh,Triangles,NULL,NULL,quiet);
	
	for (i=0;i<len;i++) free(Parts[i]);
	free(Parts);
}

int consistent_refinement(double* Refine_data,int data_size,double* H){
	const double tol = 0.1;
	const double H_thr = 0.5;
	static int flag = 0;			// used in order to force remeshing only once
	
	int i,j,k;
	int n = glob_mesh.size;
	double min_area = min_refine_area*get_total_area();
	double* R = interpolated(Refine_data,1,data_size,NULL,NULL);
	double* Error = zero_vector(n);
	
	int res = YES;
	
	if (flag==1){
		flag = 0;
		goto FINALIZE;
	}
	
	for (i=0;i<n;i++) if (H[i]<H_thr){
		if (R[i]>min_area*(1.+tol)){
			res = NO;
			Error[i] = 1.;		
			//printf("not consistent at (%f,%f): H=%f, R = %e\n",glob_mesh.Points[i].x,glob_mesh.Points[i].y,H[i],R[i]);
			
		}
		for (j=0;j<glob_mesh.Sizes[i];j++){
			k = glob_mesh.Connections[i][j];
			if (R[k]>min_area*(1.+tol)){
				res = NO;
				Error[i] = 1.;		
				//printf("not consistent at (%f,%f): H=%f, R = %e\n",glob_mesh.Points[k].x,glob_mesh.Points[k].y,H[k],R[k]);		
			}
		}
	}	
	
	FINALIZE:
	free(R);
	free(Error);
	return res;
}

int large_constraint_error(double* X){
	double e = get_constraint_work(X);
	if (beta>0 && e>eps_constraint) return 1; else return 0;
}

void solve(double* Initial_data,double* Globals,double time,long steps){
	int j,n,n_prev,accepted,last_accepted,loc_iter;
	long i;
	double u;
	double* Prev;
	double* X;
	double* M;
	double* V;
	double* New_X;
	double* Old_X;
	long total_maxiter; 
	double* Refine_data;
	
	mesh2D new_mesh;
	element_collection new_elements;
	
	// main loop settings
	loc_iter = 0;
	glob_time = time_offset;
	time_steps = 1;
	glob_time_index = 0;
	total_maxiter = (long)10*steps;
	dt_max = (double)((long double)time/steps);
	dt_min = dt_max*time_adaptive_min_dt;
	dt = dt_max;
	n = glob_mesh.size;
	if (!backtracking) energy_tol = 0;
	
	// Reflective Newton settings;
	R_Newton_Delta = sqrt((double)n)/2.;
	R_Newton_trust_radius_max = 2.*R_Newton_Delta;
	RN_set_params(R_Newton_max_iter,R_Newton_eps,R_Newton_sigma_l,R_Newton_sigma_u,R_Newton_rho,R_Newton_Delta,
	R_Newton_Xi,R_Newton_trust_lower,R_Newton_trust_upper,R_Newton_trust_radius_min,R_Newton_trust_radius_max);
	
	// set initial conditions
	Glob_mesh_sizes = (int*)malloc(sizeof(int));
	Glob_mesh_sizes[0] = glob_mesh.size;
	All_U = (double**)malloc(sizeof(double*));
	All_H = (double**)malloc(sizeof(double*));
	All_C = (double**)malloc(sizeof(double*));
	All_T = (double**)malloc(sizeof(double*));
	All_U[0] = clone_vector(Initial_data,dimension*n);									// if U is at index 0
	All_H[0] = clone_vector(&(Initial_data[dimension*n]),n);				
	All_C[0] = clone_vector(&(Initial_data[(dimension+1)*n]),n);
	All_T[0] = clone_vector(&(Initial_data[(dimension+3)*n]),n);
	
	// compute chemical potential
	V = set_vector_Ui_0_2D(1);
	M = chemical_potential(All_U[0],All_H[0],All_C[0],All_T[0]);
	vector_pseudo_div(M,V,n);
	copy_vector_content(M,Initial_data,0,(dimension+2)*n,n);
	free(M);
	free(V);
	
	// initialize global observables
	Times = (double*)malloc(sizeof(double));
	dissipation_damage = (double*)malloc(sizeof(double));
	dissipation_chemical =(double*)malloc(sizeof(double));
	heat_bound_flux = (double*)malloc(sizeof(double));
	energy_constr_lower = (double*)malloc(sizeof(double));
	energy_constr_upper = (double*)malloc(sizeof(double));
	entropy_production = (double*)malloc(sizeof(double));
	Times[0] = glob_time;
	dissipation_damage[0] = 0;
	dissipation_chemical[0] = 0;
	heat_bound_flux[0] = 0;
	energy_constr_lower[0] = -energy_tol;
	energy_constr_upper[0] = +energy_tol;
	entropy_production[0] = 0;
	free0 = Total_free_energy(All_U[0],All_H[0],All_C[0],All_T[0]);
	entropy0 = Total_entropy(All_U[0],All_H[0],All_C[0],All_T[0]);
	inner0 = Inner_Energy(All_U[0],All_H[0],All_C[0],All_T[0]);
	
	Globals[0] = Free_energy_elastic(All_U[0],All_H[0],All_C[0],All_T[0]);
	Globals[1] = Free_energy_damage(All_H[0],All_C[0]);
	Globals[2] = Free_energy_chemical(All_C[0],All_T[0]);
	Globals[3] = Free_energy_heat(All_C[0],All_T[0]);
	Globals[4] = dissipation_damage[0];
	Globals[5] = dissipation_chemical[0];
	Globals[6] = heat_bound_flux[0];
	Globals[7] = energy_constr_lower[0];
	Globals[8] = energy_constr_upper[0];
	Globals[9] = 0;
	Globals[10] = entropy_production[0];
	Globals[11] = 0;
	Globals[12] = 0;
	Globals[13] = 0;
	
	double* Z_Pred = NULL;
	double constr_work = 0;
	
	// save initial
	FILE* Output_global = open_global();
	fprintf(Output_global,"# %ld\t%d\n",steps,global_size);
	fflush(Output_global);
	fclose(Output_global);
	save(Initial_data,Globals,global_size,glob_time_index);
	
	printf("\nstart computation ...\n");
	X = clone_vector(Initial_data,degrees_of_freedom*n);
	last_accepted = YES;
	accepted = YES;
	i = 0;
	
	
	/*double* Sol = zero_vector(n);
	RN_vs_AS(Sol);
	exit(0);*/
	
	
	
	do{
		BEGINNING_OF_LOOP:
		i++;
		
		// timestep
		if (time_adaptive==YES) adjust_dt(X,Z_Pred,&dt,dt_min,dt_max,constr_work);
		
		glob_time += dt; 
		glob_time_index++;
		
		
		// refine mesh if set
		if (space_adaptive==YES){
			
			// refinement conditions (order important);
			Refine_data = generate_vector(n,-1.);
			refine_by_gradient(X,Refine_data,degrees_of_freedom,min_refine_area,max_refine_area);
			//refine_by_damage_predictor(X,Refine_data,min_refine_area);
			refine_by_damage_value(X,Refine_data,min_refine_area/50.);		
			refine_by_strain_energy(X,Refine_data,min_refine_area/50);
			//refine_by_damage_change(glob_time_index-1,Refine_data,min_refine_area/3.);												
			//refine_by_future_value(Z_Pred,Refine_data,min_refine_area/10.);
			if (Z_Pred!=NULL){	
				free(Z_Pred);
				Z_Pred = NULL;
			}	
			
			// create mesh
			create_refined_mesh(Refine_data,&(X[dimension*n]),&(X[(dimension+1)*n]),QUIET);
			Load_mesh(glob_mesh_index,&new_mesh,&new_elements,QUIET); 
			
			// compute interpolation matrix			
			if (Interpolation!=NULL) {free_sparse(Interpolation);}			
			Interpolation = Get_interpolation_matrix(&new_mesh,&glob_mesh,&new_elements,&elements);			
						
			// resize all matrices and arrays
			n = new_mesh.size;
			n_prev = glob_mesh.size;
			if (i>1){
				FreeMesh(&glob_mesh);												// do not free base mesh
				free(elements.Elements);
			}
			clean_matrices();
			set_glob_mesh(&new_mesh,&new_elements);
			set_matrices();
			
			// interpolation
			New_X = interpolated(X,degrees_of_freedom,n_prev,NULL,NULL);
			free(X);
			X = New_X;	
			
			// ensure positive H after interpolation
			for (j=0;j<n;j++) if (X[dimension*n+j]<0) X[dimension*n+j] = 0;
							
		}
		else{
			if (Interpolation!=NULL) {free_sparse(Interpolation);}
			Interpolation = sparse_identity(glob_mesh.size);	
		}
		
		// main solve
		
		Old_X = clone_vector(X,degrees_of_freedom*n);
		loc_iter = direct_alternate_minimization(X);
		if (beta>0) constr_work = fabs(get_constraint_work(X));
		
		// test energy conservation criterion go back if time adaptive
		if (time_adaptive==YES && space_adaptive==YES && dt>dt_min){ 						
			if (constr_work>eps_constraint || loc_iter>=Alt_max_iter){														
				
				if (Alt_info==YES) printf("\n step rejected -> turn one step back\n");							
				Load_mesh(glob_mesh_index-1,&new_mesh,&new_elements,QUIET); 
				set_var_number2D(1);
				sparse_matrix* Back_Interpolation = Get_interpolation_matrix(&new_mesh,&glob_mesh,&new_elements,&elements);		
				sparse_matrix* Left = get_matrix_Aij_00_2D(&new_mesh,0,0,&insert_AV);
				Z_Pred = interpolated(&(X[n*dimension]),1,n,Left,Back_Interpolation);			
				double* Y = interpolated(&(Old_X[n*(dimension+2)]),1,n,Left,Back_Interpolation);				
				free_sparse(Back_Interpolation);
				free_sparse(Left);
			
				clean_matrices();
				if (glob_mesh_index>1 || load_mode==YES){
					FreeMesh(&glob_mesh);											
					free(elements.Elements);
				}
				set_glob_mesh(&new_mesh,&new_elements);
				set_matrices();	
				n = glob_mesh.size;	
			
				X = (double*)realloc(X,degrees_of_freedom*n*sizeof(double));
				copy_vector_content(All_U[glob_time_index-1],X,0,0,dimension*n);
				copy_vector_content(All_H[glob_time_index-1],X,0,dimension*n,n);
				copy_vector_content(All_C[glob_time_index-1],X,0,(dimension+1)*n,n);
				copy_vector_content(All_T[glob_time_index-1],X,0,(dimension+3)*n,n);
				copy_vector_content(Y,X,0,(dimension+2)*n,n);
				free(Y);
				free(Old_X);
				free(Refine_data);
			
				glob_mesh_index--;
				glob_time_index--;
				glob_time -= dt;
				goto BEGINNING_OF_LOOP;	
			}
			else free(Refine_data);
		}
		else{
			if (space_adaptive==YES) printf("\nWarning: time step at minimum -> no further refinement\n");
			
		}
		
		// backtracking: check energy condition
		accepted = energy_condition_fulfilled(X,Globals);
		
		if (backtracking && !accepted){
			if (space_adaptive==YES){
				Load_mesh(glob_mesh_index-2,&new_mesh,&new_elements,QUIET); 
				set_var_number2D(1);
				sparse_matrix* Left = get_matrix_Aij_00_2D(&new_mesh,0,0,&insert_AV);
				sparse_matrix* Back_Interpolation = Get_interpolation_matrix(&new_mesh,&glob_mesh,&new_elements,&elements);
				double* Z = interpolated(&(X[n*dimension]),1,n,Left,Back_Interpolation);
				double* Y = interpolated(&(Old_X[n*(dimension+2)]),1,n,Left,Back_Interpolation);
				free_sparse(Back_Interpolation);
				free_sparse(Left);
			
				clean_matrices();
				if (glob_mesh_index>1){
					FreeMesh(&glob_mesh);											
					free(elements.Elements);
				}
				set_glob_mesh(&new_mesh,&new_elements);
				set_matrices();	
				n = glob_mesh.size;	
			
				X = (double*)realloc(X,degrees_of_freedom*n*sizeof(double));
				copy_vector_content(All_U[glob_time_index-2],X,0,0,dimension*n);
				copy_vector_content(Z,X,0,dimension*n,n);
				copy_vector_content(All_C[glob_time_index-2],X,0,(dimension+1)*n,n);
				copy_vector_content(Y,X,0,(dimension+2)*n,n);
				copy_vector_content(All_T[glob_time_index-2],X,0,(dimension+3)*n,n);
				free(Y);
				free(X);
				free(Old_X);
				
				glob_mesh_index -= 2;
			}
			
			glob_time_index -= 2;
			glob_time = Times[glob_time_index];
			goto BEGINNING_OF_LOOP;	
		}
		if (accepted && !last_accepted){
			u = dissipation_damage[glob_time_index]+Total_free_energy(&(X[0]),&(X[dimension*n])
			 ,&(X[(dimension+1)*n]),&(X[(dimension+3)*n]))-free0;
			energy_constr_lower[glob_time_index] = u-energy_tol;
			energy_constr_upper[glob_time_index] = u+energy_tol;
		}
		last_accepted = accepted;
		free(Old_X);
		
		// realloc and copy to global memory
		if (glob_time_index>=(int)time_steps){
			time_steps = glob_time_index+1;
			Times = (double*)realloc(Times,time_steps*sizeof(double));
			All_U = (double**)realloc(All_U,time_steps*sizeof(double*));
			All_H = (double**)realloc(All_H,time_steps*sizeof(double*));
			All_C = (double**)realloc(All_C,time_steps*sizeof(double*));
			All_T = (double**)realloc(All_T,time_steps*sizeof(double*));
			Glob_mesh_sizes = (int*)realloc(Glob_mesh_sizes,time_steps*sizeof(int));
		}
		Times[glob_time_index] = glob_time;
		All_U[glob_time_index] = clone_vector(&(X[0]),dimension*n);
		All_H[glob_time_index] = clone_vector(&(X[dimension*n]),n);
		All_C[glob_time_index] = clone_vector(&(X[(dimension+1)*n]),n);
		All_T[glob_time_index] = clone_vector(&(X[(dimension+3)*n]),n);
		Glob_mesh_sizes[glob_time_index] = n;
		
		// delete older values if no backtracking enabled
		if (glob_time_index>1){
			free(All_U[glob_time_index-2]);
			free(All_H[glob_time_index-2]);
			free(All_C[glob_time_index-2]);
			free(All_T[glob_time_index-2]);
			All_U[glob_time_index-2] = NULL;
			All_H[glob_time_index-2] = NULL;
			All_C[glob_time_index-2] = NULL;
			All_T[glob_time_index-2] = NULL;
		}
		
		// save
		if (glob_time_index % sample_interval == 0){			
			save(X,Globals,global_size,glob_time_index);
		}
		else save(NULL,Globals,global_size,glob_time_index);		
		
		// runtime info
		printf("\rtime: %.15f max: %f",glob_time,time+time_offset);		
		fflush(stdout);		
		if (SIGINT_flag) break;	
		
	}while(glob_time>=0 && glob_time<time+time_offset && i<total_maxiter);
	
	if (glob_time==0) printf("\nfinished: returned to t=0\n\n");
	else if (glob_time>=time+time_offset) printf("\nfinished: max time reached\n\n");
	else if (i>=total_maxiter) printf("\naborted: no convergence\n\n");
	
	// clean
	FreeMesh(&glob_mesh);
	free(elements.Elements);
	free_sparse(Interpolation);
	for (i=0;i<(int)time_steps;i++){
		if (All_U[i]!=NULL) free(All_U[i]);
		if (All_H[i]!=NULL) free(All_H[i]);
		if (All_C[i]!=NULL) free(All_C[i]);
		if (All_T[i]!=NULL) free(All_T[i]);
	}
	free(All_U);
	free(All_H);
	free(All_C);
	free(All_T);
	free(X);
	free(dissipation_damage);
	free(dissipation_chemical);
	free(heat_bound_flux);
	free(energy_constr_lower);
	free(energy_constr_upper);
	free(entropy_production);
}

void smoother(sparse_matrix* A,double* F,double* Sol,int deg_freedom,int iter){
	int i,j,k,l,I,ind,diag,bottom,top,d,mesh_ind;							
	const int chunk_size = 5000;
	double sum;
	int n = A->size;
	int N = n/deg_freedom;
	double* Old = zero_vector(n);
	
	
	for (k=0;k<iter;k++){		
		copy_vector_to(Sol,Old,n);
		#pragma omp parallel private(i,j,d,I,sum,diag,ind)
		{
			#pragma omp for schedule(dynamic,chunk_size) 
			for (i=0;i<N;i++){
				for (d=0;d<deg_freedom;d++){
					I = i+N*d;
					
						sum = F[I];
						diag = -1;
						for (j=0;j<A->Len[I];j++){
							ind = A->Indices[I][j];
							if (ind!=I) sum -= A->Values[I][j]*Old[ind];
							else diag = j;
						}
						//#pragma omp critical 
						Sol[I] = sum/A->Values[I][diag];
										
				}
				//Old[i] += (double)i/N;				
			}
		}
	}
	free(Old);		
}

///////////////////////////// test functions /////////////////////////////////////////

int solve_CH(double* C){
	int i;
	double res,res0;
	double* b;
	sparse_matrix* A;
	sparse_matrix* A_nonlin;
	umfpack_matrix_info A_info;
	
	int n = glob_mesh.size;
	void* A_LU_info = NULL;
	double* Y = generate_vector(n,1.);
	double* H = zero_vector(n);
	double* T = zero_vector(n);
	sparse_matrix* A_lin = get_CH_linear_part(H,Y,T);
	double* X0 = zero_vector(2*n);
	double* X = zero_vector(2*n);
	copy_vector_content(C,X,0,0,n);
	double* X_prev = clone_vector(X,2*n);
	
	A = clone(A_lin);
	A_nonlin = get_CH_nonlinear_part(NULL,X_prev,NULL);
	sparse_add(A,A_nonlin,-1.);
	if (multifrontal_solver_get_LU(A,&A_LU_info,&A_info)==FAILED) return FAILED;
	
	res0 = 1.;
	i = 0;
	do{
		copy_vector_content(X,X0,0,0,2*n);
		
		// right side
		b = Jacobian_self_c(X,NULL);													// already multiplied by Aij00
		linear_map(b,-1.,A_nonlin,X),
		linear_map(b,1.,Phase_ID_c,X_prev);
		
		// solve
		multifrontal_solver_solve_LU(A_LU_info,&A_info,X,b);
		free(b);
		
		res = vec_dist(X,X0,2*n);		
		if (res>res0){
			free_sparse(A_nonlin);
			free_sparse(A);
			free_UMFPACK_matrix_info(&A_info);
			umfpack_di_free_numeric(&A_LU_info);
			A = clone(A_lin);
			A_nonlin = get_CH_nonlinear_part(NULL,X,NULL);	
			sparse_add(A,A_nonlin,-1.);			
			if (multifrontal_solver_get_LU(A,&A_LU_info,&A_info)==FAILED) return FAILED;
		}else res0 = res;
		i++;
	}while(res>Alt_eps && i<Alt_max_iter);
	if (isnan(res)){
		printf("solution diverged -> abort\n");
		exit(0);
	}
	printf("solve C-H: residuum %e iterations %d\n",res,i);
	copy_vector_content(X,C,0,0,n);
	
	free_UMFPACK_matrix_info(&A_info);
	umfpack_di_free_numeric(&A_LU_info);
	free_sparse(A);
	free_sparse(A_lin);
	free_sparse(A_nonlin);
	free(X_prev);
	free(X0);
	free(X);
	free(H);
	free(T);
	free(Y);
	
	if (i<Alt_max_iter) return SUCCESS; else return FAILED;
}

void test_CH(){
	int i;
	double x,y;
	char Fullname[512];

	Alt_max_iter = 1000;
	int n = glob_mesh.size;
	double* C = zero_vector(n);
	for (i=0;i<n;i++){
		x = glob_mesh.Points[i].x;
		y = glob_mesh.Points[i].y;
		C[i] = 0.01*sin(8.*M_PI*(x+y));				
	}
	
	sprintf(Fullname,"%s/%s/%s%d",Output_dir,Output_name,"phase",0);
	print_scalar_data(Fullname,C,n);
	for (i=1;i<=time_steps;i++){
		solve_CH(C);
		if (i % sample_interval==0){
			sprintf(Fullname,"%s/%s/%s%d",Output_dir,Output_name,"phase",i);
			print_scalar_data(Fullname,C,n);
		}
	}
	printf("finished\n");
	exit(0);
}

double Fx(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	return cos(4.*x)-1.-y*y;		
}
	
double Fy(double* X,int k){
	double x = glob_mesh.Points[k].x;
	double y = glob_mesh.Points[k].y;
	return -4.*y*y;		
}

double f(double x){
	return -(x-1)*(x-1)+(x-1)*(x-1)*(x-1)*(x-1);
}

void reference_disc(double* Lower,double* Upper,point2D* Pos,point2D* Shift){
	int i;
	int n = glob_mesh.size;
	int m = get_center_index();
	Shift->x = Pos->x;
	Shift->y = Pos->y;
	vec_add_mult(Shift,&(glob_mesh.Points[m]),-1.);
	for (i=0;i<n;i++){
		if (!isnan(Lower[i])) Lower[i] -= glob_mesh.Points[i].x+Shift->x;
		if (!isnan(Lower[i+n])) Lower[i+n] -= glob_mesh.Points[i].y+Shift->y;
		
		if (!isnan(Upper[i])) Upper[i] -= glob_mesh.Points[i].x+Shift->x; 
		if (!isnan(Upper[i+n])) Upper[i+n] -= glob_mesh.Points[i].y+Shift->y;
	}
}

void elastic_energy_1D(sparse_matrix* A,double* b,double L){
	int i;
	int n = A->size;
	double h = (double)L/(n-1);
	
	insert_sparse(A,1./h,0,0);
	insert_sparse(A,-1./h,0,1);
	insert_sparse(A,1./h,n-1,n-1);
	insert_sparse(A,-1./h,n-1,n-2);
	b[0] = h/2.;
	b[n-1] = h/2.;
	
	for (i=1;i<n-1;i++){
		insert_sparse(A,2./h,i,i);
		insert_sparse(A,-1./h,i,i-1);
		insert_sparse(A,-1./h,i,i+1);
		b[i] = h;
	}
}

void test_product(){
	
	int num_fields = 2;
	set_var_number(num_fields);	
	sparse_matrix* A = set_matrix_Aij_11_2D(0,0,&insert_A_bbV_a);
	sparse_matrix* B = set_matrix_Aij_11_2D(0,0,&insert_A_abV_b);
	sparse_matrix* C = sparse_product(A,B);
	sparse_matrix* FC;
	//printf("B symmetric ? %d\n")
	product_pattern pattern = set_product_pattern(A,B,NULL,&FC);
	
	int i;
	int N = 10;
	set_thread_num(8);
	
	printf("start...\n");
	fflush(stdout);
	time_t start,fin;
	time(&start);
	
	
	
	for (i=0;i<N;i++){
		fast_sparse_product(FC,A,B,&pattern);
		//sparse_matrix* FC = sparse_product(A,B);	
	}
	
	time(&fin);
	printf("physical execution time: %d sec\n",(int)(fin-start));
	
	
	exit(0);
}

void test_BiCGStab(){
	
	int i;
	int num_fields = 2;
	set_var_number(num_fields);	
	set_thread_num(1);
	set_omp_chunk_size(100);
	
	//sparse_matrix* A = read_sparse_textfile("/Home/damage/radszuwe/Daten","matrix");
	int n = glob_mesh.size;
	set_var_number2D(2);
	sparse_matrix* A = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_abV_b);
	scalar_sparse_mult(lambda_1,A);
	sparse_matrix* A2 = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_baV_b);
	sparse_matrix* A3 = set_matrix_Aij_11_2D(n_u,n_u,&insert_A_bbV_a);
	sparse_add(A,A2,mu_1);
	sparse_add(A,A3,mu_1);
	free_sparse(A2);
	free_sparse(A3);
		
	sparse_matrix* B3 = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_abV_b);
	scalar_sparse_mult(lambda_1,B3);
	sparse_matrix* B1 = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_baV_b);	
	sparse_matrix* B2 = set_matrix_bound_2_Aij_10_2D(0,0,&insert_A_bbV_a);
	sparse_add(B3,B1,mu_1);
	sparse_add(B3,B2,mu_1);
	free_sparse(B1);
	free_sparse(B2);
	sparse_matrix* B = get_transpose(B3,2*n);
		
	sparse_matrix* Tx;
	sparse_matrix* Ty;
	sparse_matrix* Trow_x = set_matrix_Aij_01_2D(0,0,&insert_A_cV_cE_ab_row_x);
	sparse_matrix* Trow_y = set_matrix_Aij_01_2D(0,0,&insert_A_cV_cE_ab_row_y);
	scalar_sparse_mult(lambda_1,Trow_x);
	scalar_sparse_mult(lambda_1,Trow_y);
	
	Tx = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_x);
	Ty = set_matrix_Aij_01_2D(0,0,&insert_A_aV_b_row_y);
	sparse_add(Trow_x,Tx,mu_1);
	sparse_add(Trow_y,Ty,mu_1);
	free_sparse(Tx);
	free_sparse(Ty);
	
	Tx = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_x);
	Ty = set_matrix_Aij_01_2D(0,0,&insert_A_bV_a_row_y);
	sparse_add(Trow_x,Tx,mu_1);
	sparse_add(Trow_y,Ty,mu_1);
	free_sparse(Tx);
	free_sparse(Ty);
	
	double* b = zero_vector(2*n);
	glob_time = 1.;
	
	bound_cond* BC = get_U_bound_conditions(glob_time);
	Set_dirichlet_BC(A,b,BC);
	
	set_system_matrix(A);
	BI_set_mat_mult(&matvec);
	BI_set_AMG_preconditioner(A,2,3);
	
	time_t start,fin;
	time(&start);
	
	double* X = zero_vector(2*n);
	printf("start solver...\n");
	fflush(stdout);	
	i = BiCGStab_solve(X,b,2*n);
	
	printf("iterations: %d\n",i);
	
	time(&fin);
	printf("physical execution time: %d sec\n",(int)(fin-start));
	fflush(stdout);
	
	double eps = 1e-10;
	double* Fx = sparse_mult(Trow_x,X);
	double* Fy = sparse_mult(Trow_y,X);
	double* Txx = compute_scalar_function(&(Fx[0]),YES,eps);
	double* Txy = compute_scalar_function(&(Fx[n]),YES,eps);
	double* Tyx = compute_scalar_function(&(Fy[0]),YES,eps);
	double* Tyy = compute_scalar_function(&(Fy[n]),YES,eps);
	
	double* dB = sparse_mult(B,X);	
	
	print_scalar_data("/Home/damage/radszuwe/Daten/scalardata",&(dB[n]),n);
	print_vector(X,2*n);
	
	exit(0);
	
}


void Test_AMG(sparse_matrix* A){
	
	int depth,smiter;
	int num_fields = dimension;
	int n = glob_mesh.size;
	sparse_matrix* K = clone(A);
	double* b = zero_vector(dimension*n);
	
	bound_cond* BC = get_U_bound_conditions(glob_time);
	Set_dirichlet_BC(K,b,BC);
	
	AMG_set_max_iter(1000);
	AMG_print_info(YES);
	AMG_set_SOR_relaxation_coeff(1.6);
	AMG_set_matrix_tolerance(1e-5);
	smiter = 2;
	depth = 5;
	
	double* X = zero_vector(dimension*n);
	double r0 = matrix_residuum(K,X,b);
	
	clock_t start = clock();
	amg_system_info* Setup_data = AMG_setup(K,num_fields,depth,smiter,smiter+1,AGGRESSIVE,STAND_ALONE,STANDARD,NULL,NULL);
	AMG_solve(Setup_data,b,X,NULL);
	clock_t end = clock();
	
	double r = matrix_residuum(K,X,b);
	
	printf("residuum reduction: %e\n",r/r0);
	printf("\ntime: %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	exit(0);
	
}

void test_AMG(){
	
	int i,depth,smiter;
	int num_fields = 1;
	set_var_number(num_fields);	
	
	sparse_matrix* AMG_sparse = read_sparse_textfile("/Home/damage/radszuwe/Daten","ugly_matrix");
	//sparse_matrix* AMG_sparse = set_matrix_Aij_11_2D(0,0,&insert_A_bbV_a);
	//sparse_matrix* Tr = set_matrix_Aij_11_2D(0,0,&insert_A_abV_b);
	//sparse_add(AMG_sparse,Tr,0.3);
	int n = AMG_sparse->size;
	
	
	
	
	int s = 20;
	/*sparse_matrix* H = NULL;
	double** U = (double**)malloc(s*sizeof(double*));
	for (i=0;i<s;i++) U[i] = NULL;
	double* V0 = generate_vector(n,1.);
	reortho_Arnoldi(AMG_sparse,&H,U,V0,&s,1);
	
	double* Eig = zero_vector(s);
	double** V = (double**)malloc(s*sizeof(double*));
	for (i=0;i<s;i++) V[i] = NULL;
	Ritz_values(H,V,Eig,200);*/
	
	
	double** W = (double**)malloc(s*sizeof(double*));
	/*for (i=0;i<s;i++){
		int j;
		W[i] = zero_vector(n);
		for (j=0;j<s;j++) vector_add(W[i],U[j],V[i][j],n);
		vector_normalize(W[i],n);
	}*/
	
	double* Eig = zero_vector(s);
	double* W0 = generate_vector(n,1.);
	Harmonic_Ritz_values(AMG_sparse,W,W0,Eig,s);
	
	sparse_matrix* ID = sparse_identity(n);
	for (i=0;i<s;i++){
		sparse_matrix* D = clone(AMG_sparse);
		sparse_add(D,ID,-Eig[i]);
		double* Z = sparse_mult(D,W[i]);
		printf("eigen value: %e\n",Eig[i]);
		printf("eigen residuum %d: %e\n",i,euklid_norm(Z,n)/matrix_norm(D));
		free_sparse(D);
		free(Z);
	}
	
	double eig= 1.;
	double* Vec = zero_vector(AMG_sparse->size);
	i = get_most_negative_eigenvector(AMG_sparse,&eig,Vec,10000,1e-10);
	printf("iterations: %d, smallest eigenvalue: %e\n",i,eig);
	
	vector_add(Vec,W[0],-1.,n);
	printf("dist to smallest EV: %e\n",euklid_norm(Vec,n));
	
	/*exit(0);
	free(W[s-1]);
	W[s-1] = clone_vector(Vec,n);
	Eig[s-1] = eig;*/
	
	sparse_matrix* QT = sparse_zero(s);
	convert_array_to_sparse(W,QT,s,n);
	sparse_matrix* Q = get_transpose(QT,n);
	sparse_matrix* AQ = sparse_product(AMG_sparse,Q);
	sparse_matrix* QTAQ = sparse_product(QT,AQ);
	print_sparse(QTAQ);	
	sparse_matrix* LA = sparse_zero(s);
	sparse_matrix* UA = sparse_zero(s);
	complete_LU_factorization(QTAQ,LA,UA);
	
	
	
	//index2D ind = get_most_distinct_points();
	//int Constr[4] = {ind.i,ind.i+glob_mesh.size,ind.j,ind.j+glob_mesh.size};
	
	int** AMG_cond = set_pure_conditions(num_fields,DIRICHLET);
	//sparse_matrix* AMG_Bound = FEM_get_dirichlet_correction(AMG_sparse,AMG_cond,1.);
	double* b = get_function(NULL,glob_mesh.size,num_fields,0,&Fx);
	//double* by = get_function(NULL,glob_mesh.size,num_fields,1,&Fy);			only when 2D
	//vector_add(b,by,1.,n);
	double* B = set_vector_Ui_0_2D(num_fields);
	double* Bound_val = zero_vector(n);
	vector_pseudo_mult(b,B,n);
	//linear_map(b,1.,AMG_Bound,Bound_val);
	//include_aux_constraints(AMG_sparse,b,Constr,4);
	
	AMG_set_max_iter(5);
	AMG_print_info(YES);
	AMG_set_SOR_relaxation_coeff(1.6);
	smiter = 2;
	depth = 4;
	
	amg_system_info* Setup_data = AMG_setup(AMG_sparse,num_fields,depth,smiter,smiter+1,AGGRESSIVE,STAND_ALONE,DIRECT,NULL,NULL);
	double* X = zero_vector(n);
	
	
	
	double* Y = zero_vector(n);

	clock_t start = clock();
	
	multifrontal_solver(AMG_sparse,Y,b,"",NULL);
	
	clock_t end = clock();
	
	printf("umfpack residuum: %e\n",matrix_residuum(AMG_sparse,Y,b));
	printf("\ntime: %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	
	start = clock();
	
	double r0 = matrix_residuum(AMG_sparse,X,b);
	AMG_solve(Setup_data,b,X,NULL);
	
	for (i=0;i<10;i++){	
		double* R = clone_vector(b,n);
		linear_map(R,-1.,AMG_sparse,X);
		double* Rc = sparse_mult(QT,R);
		L_triang_invert(LA,Rc);
		U_triang_invert(UA,Rc);
		double* Dx = sparse_mult(Q,Rc);
		vector_add(X,Dx,1.,n);
		free(R);
		free(Rc);
		free(Dx);
		printf("norm X: %e\n",euklid_norm(X,n));
		AMG_solve(Setup_data,b,X,NULL);
	}
	
	
	double r = matrix_residuum(AMG_sparse,X,b);
	
	end = clock();
	printf("\ntime: %f sec\n",(double)(end-start)/CLOCKS_PER_SEC);
	printf("overall residuum: %e\n",r/r0);
	
	//double* Shift = clone_vector(X,n);//zero_vector(2*n);
	//print_vector(Shift,n);
	//print_scalar_data("/Home/damage/radszuwe/Daten/amg",X,n);
	//print_vector(sparse_row_sum(Setup_data->Prolongation[1]),(Setup_data->Prolongation[1])->size);
	exit(0);
}

void test_1D(int n){
	int i;
	double x,eig;
	double* F = zero_vector(n);
	
	for (i=0;i<n;i++){
		x = (double)i/(n-1);
		F[i] = 0.*exp(-10.0*(x-0.5)*(x-0.5));
	}
	
	//sparse_matrix* A = get_1D_Laplace(F,n);
	sparse_matrix* A = get_test_matrix(n,0);
	double* V1 = zero_vector(n);
	double* V2 = zero_vector(n);
	double* X = zero_vector(n);
	V1[0] = 1.0/sqrt(2.0);
	V1[3] = 1.0/sqrt(2.0);
	V2[1] = 1.0;
	F[1] = 0.014;
	//F[0] = -0.02;
	subspace_minimum(A,F,V1,V2,X);
	print_sparse(A);
	print_vector(X,n);
	exit(0);
	
	double* V = generate_vector(n,1./sqrt((double)n));
	printf("start computation\n");
	int iter = get_most_negative_eigenvector(A,&eig,V,100000,1e-8);
	printf("itrations: %d\n",iter);
	printf("minimal eigenvalue: %f\n",eig);
	
	free_sparse(A);
	free(F);
}

void test_arg(int argc,int n){
	if (argc<n){
		printf("too few arguments -> abort\n");
		exit(0);
	}
}

///////////////////////////////////////////// main program //////////////////////////////////////////

int main(int argc, char* argv[]){
		
	signal(SIGINT,&signal_handler);
	
	Home_dir = getenv("HOME");
	test_arg(argc,3);
	
	if (strcmp(argv[1],"sim")==0){
		time_t start,fin;
		
	
		// set default parameter directory
		sprintf(Param_dir,"%s/%s",getenv("HOME"),"Daten");
		Prog_name = argv[0];
		
		// setup and initial conditions
		double* Solution;
		setup(argc,argv,&Solution,&set_initial_conditions);				
		
		// record time
		time(&start);
		
		// main solve
		double* Globals = zero_vector(global_size);
		solve(Solution,Globals,total_time,time_steps);
		
		// print time
		time(&fin);
		printf("physical execution time: %d sec\n",(int)(fin-start));
		printf("umfpack time:\n");
		printf("\t damage: %f sec \n",UMFPACK_time_damage);
		printf("\t chemistry: %f sec \n",UMFPACK_time_chemistry);
		printf("\t temperature: %f sec \n",UMFPACK_time_temperature);
		printf("\t elasticity: %f sec \n",UMFPACK_time_elastic);
		printf("\t interpolation: %f sec \n",UMFPACK_time_interpolation);
		printf("\t sum: %f sec \n",UMFPACK_time_damage+UMFPACK_time_chemistry
		 +UMFPACK_time_temperature+UMFPACK_time_elastic+UMFPACK_time_interpolation);
		fflush(stdout);
		
		//print numerical statistics		
		print_RN_statistics(stdout);
		AMG_print_stat(&alter_min_stat);
		printf("condition number: min=%e max=%e\n",min_cond_num,max_cond_num);
				
		// clean
		free(Solution);
		free(Globals);
		//clean();
	}
	if (strcmp(argv[1],"mesh")==0){						
		
		double a;
		char Command[512];
		
		int attribute = NO;
		test_arg(argc,3);
		
		// mesh creation
		
		char FullName[512];
		char* CondName = NULL;
		char* Dir = "/Home/damage/radszuwe/Daten";	
		char* Name = argv[2];
		
		int b_num = atoi(argv[3]);
		sprintf(FullName,"%s/%s",Dir,Name);
		printf("create mesh data in file %s\n",FullName);
		strcat(FullName,".poly");
		
		if (argc==5 && strcmp(argv[4],"-cond")==0){
			CondName = (char*)malloc(512*sizeof(char));
			sprintf(CondName,"%s/%s.con",Dir,Name);			
			printf("create condition file %s\n",CondName);
		}		
		
		if (attribute==NO){
			//a = create_shape_from_image(FullName,b_num);						// wias logo
			//a = create_2D_mesh(FullName,b_num);								// disc
			//a = create_shape1a(FullName,b_num);								// carved rectangle
			//a = create_shape2(FullName,b_num);								// rectangle
			//a = create_shape3(FullName,b_num);								// rectangle with circular holes			
			//a = create_shape4(FullName,b_num);								// notch
			//a = create_shape5(FullName,b_num);								// two overlapping squares
			//a = create_shape6a(FullName,b_num);								// rectangle with one thin notch
			//a = create_shape7(FullName,b_num);								// solder drop between plates
			//a = create_shape8(FullName,b_num);								// rectangle with rectangular holes
			a = create_shape_hexagons(FullName,CondName,b_num);
			//a = create_shape9(FullName,b_num);								// wheel 
			sprintf(Command,"triangle -p -v -q30 -a%f %s",a,FullName);
		}
		else{
			a = create_attr1(FullName,b_num);
			sprintf(Command,"triangle -p -v -q30 -a%f -A -X %s",a,FullName);
		}
		
		if (CondName!=NULL) free(CondName);
		
		printf("execute: %s\n",Command);
		system(Command);
		
		if (argc==5 && strcmp(argv[4],"-refine")==0){
			char NewName[512];
			sprintf(NewName,"%s.1",Name);	
			read_mesh_2D(Dir,NewName,NULL,NULL,NULL,NULL,CHATTY);					
			sprintf(FullName,"%s/%s.1",Dir,Name);	
			
			//F_area_bound_curvature(NULL);									// init curvature function				
			refine_mesh(FullName,&F_area_shape6,CHATTY);		
		}		
	}
	
	return 0;
}
