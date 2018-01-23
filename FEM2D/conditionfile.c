#include "conditionfile.h"

double linear(double x,double period,double min,double max){
	return min+(max-min)*x/period;	
}

double saw_signal(double x,double period,double offset,double min,double max){
	x -= offset;
	double n = floor(x/period);
	return min+(max-min)*(x-n*period)/period;
}

double rect_signal(double x,double period,double offset,double min,double max){
	x -= offset;
	double n = floor(x/period);
	double s = (x-n*period)/period;
	return (s<0.5) ? min : max;
}

double step_signal(double x,double x_c,double dx,double val_before,double val_after){
	if (x<x_c-dx/2.) return val_before;
	else if (x>x_c+dx/2.) return val_after;
	else return val_before+(val_after-val_before)*(x-x_c+dx/2.)/dx;
}

double linear_updown_signal(double x,double period,double offset,double min,double max){
	x -= offset;
	double n = floor(x/period);
	double s = (x-n*period)/period;
	return (s<0.5) ? min+2.*(max-min)*s : min+2.*(max-min)*(1.-s);
}

double ramp_signal(double x,double period,double offset,double min,double max){
	return min+(max-min)*rect_signal(-x,period,offset,0,1.)*saw_signal(x,period/2.,offset,0,1.);
	
	/*const double amplitude_stretch = 0.01;
	const double amplitude_shear = 0.05;
	const double T = 0.01;
	
	if (x>0){
		if (var_index==0){
			return (sin(M_PI*t/T)>0) ? amplitude_stretch*saw(t,T) : 0;
		}
		if (var_index==1){
			return (sin(M_PI*t/T)<=0) ? amplitude_shear*saw(t,T) : 0;
		}
		return 0;
	}
	else return 0;*/
}

double linear_saturation(double x,double xmax,double min,double max){
	return (x<xmax) ? linear(x,xmax,min,max) : max;
}

point2D* parse_point2D(char* part){
	char Buffer[256];
	strcpy(Buffer,&(part[1]));
	
	char* X = strtok(Buffer,"(,)");
	if (X==NULL) return NULL;
	
	char* Y = strtok(NULL,"(,)");
	if (Y==NULL) return NULL;
	
	point2D* P = (point2D*)malloc(sizeof(point2D));
	P->x = atof(X);
	P->y = atof(Y);
	return P;
}

int parse_cond(char* Part,int* cond,char* time_depend){
	
	char Buffer[256];
	strcpy(Buffer,Part);
	char* sub = strtok(Buffer,"()");
	if (sub!=NULL){
		if (strcmp(sub,"DIRICHLET")==0) *cond = DIRICHLET;
		else if (strcmp(sub,"NEUMANN")==0) *cond = NEUMANN;
		else if (strcmp(sub,"ROBIN")==0) *cond = ROBIN;
		else *cond = NOCON;
	}
	else return FAILED;
	
	sub = strtok(NULL,"()");
	if (sub!=NULL) sprintf(time_depend,"%s",sub);
	else return FAILED;
	
	return SUCCESS;
}

int parse_bc_line(char* line,bound_info* info,int max_len){
	int m,flag,pos;
	point2D* P;
	
	int l = (info->loops)-1;
	info->Sizes[l] = 0;
	info->Coords[l] = NULL;
	info->Cond[l] = NULL;
	info->Time_depend[l] = NULL;
	
	char* Part = strtok(line,"\t");
	pos = strlen(Part)+1;
	
	flag = 0;
	while(Part!=NULL){
		if (!flag){
			P = parse_point2D(Part);
			if (P==NULL) return FAILED;
			m = (info->Sizes[l]);
			(info->Sizes[l])++;
			info->Coords[l] = (point2D*)realloc(info->Coords[l],(m+1)*sizeof(point2D));
			info->Coords[l][m] = init_point2D(P->x,P->y);			
			flag = 1;			
			free(P);
		}
		else{
			info->Cond[l] = (int*)realloc(info->Cond[l],(m+1)*sizeof(int));
			info->Time_depend[l] = (char**)realloc(info->Time_depend[l],(m+1)*sizeof(char*));
			info->Time_depend[l][m] = (char*)malloc(256*sizeof(char));
			if (parse_cond(Part,&(info->Cond[l][m]),info->Time_depend[l][m])==FAILED) return FAILED;
			flag = 0;
		}
		
		if (pos<max_len){
			Part = strtok(&(line[pos]),"\t");
			if (Part!=NULL) pos += strlen(Part)+1;
		}
		else Part = NULL;		
	};
	
	if (flag!=0) return FAILED; else return SUCCESS;
}

int process_bound(bound_info** Bound,FILE* file){
	int status,len,var;
	char* Var;
	char* Buffer = NULL;
	
	do{
		size_t buff_len = 0;
		if (Buffer!=NULL) free(Buffer);
		Buffer = NULL;
		len = 0;
		status = getline(&Buffer,&buff_len,file);
		len = strlen(Buffer);
		//printf("%s\n",Buffer);
		if (status>0 && Buffer[0]!='#' && strstr(Buffer,"var")!=NULL){
			Var = strtok(Buffer," ");
			Var = strtok(NULL," ");
			var = atoi(Var); 
			Bound[var] = (bound_info*)malloc(sizeof(bound_info));
			Bound[var]->loops = 0;
			Bound[var]->Sizes = NULL;
			Bound[var]->Cond = NULL;						
			Bound[var]->Coords = NULL;
			Bound[var]->Time_depend = NULL;	
		}
		if (status>1 && Buffer[0]!='#' && strstr(Buffer,"var")==NULL && strstr(Buffer,"end")==NULL){
			(Bound[var]->loops)++;
			Bound[var]->Sizes = (int*)realloc(Bound[var]->Sizes,(Bound[var]->loops)*sizeof(int));
			Bound[var]->Coords = (point2D**)realloc(Bound[var]->Coords,(Bound[var]->loops)*sizeof(point2D*));
			Bound[var]->Cond = (int**)realloc(Bound[var]->Cond,(Bound[var]->loops)*sizeof(int*));
			Bound[var]->Time_depend = (char***)realloc(Bound[var]->Time_depend,(Bound[var]->loops)*sizeof(char**));
			if (parse_bc_line(Buffer,Bound[var],len)==FAILED){
				printf("weird line: %s in .con file\n",Buffer);
				return FAILED;					
			}
		}
	}while(status>=0 && strcmp(Buffer,"end(BC)\n")!=0);				
	
	return SUCCESS;
}

int process_init(init_info* Init,FILE* file){
	int status,len;
	char* Part;
	char* Buffer = NULL;
	
	Init->size = 0;
	Init->Indices = NULL;
	Init->Cond = NULL;
	do{
		if (Buffer!=NULL) free(Buffer);
		Buffer = NULL;
		size_t buff_len = 0;
		status = getline(&Buffer,&buff_len,file);
		if (status>0 && Buffer[0]!='#' && strstr(Buffer,"var")!=NULL){
			Part = strtok(Buffer," ");
			if (strcmp(Part,"var")!=0) return FAILED;
			else{
				(Init->size)++;
				Init->Indices = (int*)realloc(Init->Indices,(Init->size)*sizeof(int));
				Init->Cond = (char**)realloc(Init->Cond,(Init->size)*sizeof(char*));
				
				Part = strtok(NULL," ");
				if (Part!=NULL) Init->Indices[(Init->size)-1] = atoi(Part); else return FAILED;
				Part = strtok(NULL," ");
				Init->Cond[(Init->size)-1] = (char*)malloc(MAX_FUNC_SIZE*sizeof(char));
				if (Part!=NULL) strcpy(Init->Cond[(Init->size)-1],Part); else return FAILED;
			}				
		}
	}while(status>=0 && strcmp(Buffer,"end(IC)\n")!=0);
	
	return SUCCESS;
}

int load_condition_file(char* Fullname,bound_info** Bound,int bound_size,init_info* Init){
	int i,var,status,res;
	
	for (i=0;i<bound_size;i++) Bound[i] = NULL;
	FILE* file = fopen(Fullname,"r");
	if (file==NULL){
		printf("could not open condition file %s -> skip\n",Fullname);
		return FAILED;
	}
	else{
		int len;
		char* Var;
		char* Buffer = NULL;
		size_t buff_len = 0;
		
		do{
			status = getline(&Buffer,&buff_len,file);
			if (status>0 && Buffer[0]!='#' && strstr(Buffer,"begin")){							// comments begin with #
				if (strcmp(Buffer,"begin(BC)\n")==0) res = process_bound(Bound,file);
				else if (strcmp(Buffer,"begin(IC)\n")==0) res = process_init(Init,file);
				if (res==FAILED){
					return FAILED;
				}
			}
			
		}while(status>=0);	
		if (Buffer!=NULL) free(Buffer);
	}
	
	return SUCCESS;
}

double std_exp(char* F,double x,double y,double t){
	if (strcmp(F,"x")==0) return x;
	else if (strcmp(F,"y")==0) return y;
	else if (strcmp(F,"t")==0) return t;
	else return atof(F);
}

void region_strcpy(char* Dest,char* Source_start,char* Source_end){
	char* P = Source_start;
	while(P<Source_end){
		*Dest = *P;
		Dest++;
		P++;
	}
	*Dest = '\0';
}

void remove_brackets(char* Expr){
	int i;
	int len = strlen(Expr);
	if (len>1){
		for (i=1;i<len;i++){
			Expr[i-1] = Expr[i];
		}
		Expr[len-2] = '\0';
	}
}

int outer_brackets(char* Expr){	
	int res = 0;
	int len = strlen(Expr);
	if (Expr[0]=='(' && Expr[len-1]==')'){
		int brac = 0;
		char* Pos = Expr;
		while(Pos!=&(Expr[len-1])){
			if (*Pos=='(') brac++;
			if (*Pos==')') brac--;
			Pos = strpbrk(Pos+1,"()");	
			if (brac==0) return 0;
		}
		return 1;
	}
	else return 0;			
}

char* scan_for_plusminus(char* Start){
	int brac = 0;
	char* Res = NULL;
	char* Pos = strpbrk(Start,"+-()");	
	while(Pos!=NULL){
		switch(*Pos){
			case '+':
				if (brac==0){
					Res = Pos;
					goto FINALIZE;				
				}
				break;
			case '-':
				if (brac==0){
					Res = Pos;
					goto FINALIZE;	
				}
				break;
			case '(':
				brac++;
				break;
			case ')':
				brac--;
				break;
		}
		Pos = strpbrk(Pos+1,"+-()");	
	}
	FINALIZE: return Res;
}

char* scan_for_divmult(char* Start){
	int brac = 0;
	char* Res = NULL;
	char* Pos = strpbrk(Start,"*/()");	
	while(Pos!=NULL){
		switch(*Pos){
			case '*':
				if (brac==0){
					Res = Pos;
					goto FINALIZE;				
				}
				break;
			case '/':
				if (brac==0){
					Res = Pos;
					goto FINALIZE;	
				}
				break;
			case '(':
				brac++;
				break;
			case ')':
				brac--;
				break;
		}
		Pos = strpbrk(Pos+1,"*/()");	
	}
	FINALIZE: return Res;
}

double parse_expression2D(char* Expression,double x,double y,double t){		// expression must be null-terminated
																			// no constant string allowed
	double temp,fac;
	char* Pos;
	char* Temp;
	
	if (outer_brackets(Expression)) remove_brackets(Expression);
	double res = 0;
	double last_fac = 1.;
	int counter = 0;
	int len = strlen(Expression);
	char* Last = Expression;	
	
	
	// +,- partition
	do{
		Pos = scan_for_plusminus(Last);
		if (Pos!=NULL){
			fac = (*Pos=='+') ? 1. : -1.;
			if (Pos>Last){
				Temp = (char*)malloc(len*sizeof(char));				
				region_strcpy(Temp,Last,Pos);
				res += last_fac*parse_expression2D(Temp,x,y,t);				
				free(Temp);
			}
			Last = Pos+1;
			last_fac = fac;
			counter++;
		}
	}while(Pos!=NULL);
	Temp = (char*)malloc(max(len+1,2)*sizeof(char));
	region_strcpy(Temp,Last,&(Expression[len]));
	if (counter>0) res += last_fac*parse_expression2D(Temp,x,y,t);
	else{
		
		// *,/ partition
		char* Temp2;
		double expo;
		
		int len2 = strlen(Temp);
		double last_expo = 1.;
		counter = 0;
		res = 1.;
		Last = Temp;
		do{
			Pos = scan_for_divmult(Last);
			if (Pos!=NULL){
				expo = (*Pos=='*') ? 1. : -1.;
				Temp2 = (char*)malloc(max(len+1,2)*sizeof(char));			
				region_strcpy(Temp2,Last,Pos);
				if (last_expo==1.) res *= parse_expression2D(Temp2,x,y,t); else res /= parse_expression2D(Temp2,x,y,t);
				Last = Pos+1;
				free(Temp2);
				last_expo = expo;
				counter++;
			}
		}while(Pos!=NULL);
		Temp2 = (char*)malloc(max(len2+1,2)*sizeof(char));
		region_strcpy(Temp2,Last,&(Temp[len2]));
		if (counter>0){
			if (last_expo==1.) res *= parse_expression2D(Temp2,x,y,t); else res /= parse_expression2D(Temp2,x,y,t);	
		}
		else{
			Pos = strpbrk(Temp2,"(");
			if (Pos==NULL){
				switch(Temp2[0]){
					case 'x': res = x;break;
					case 'y': res = y;break;
					case 't': res = t;break;
					case 'p': res = M_PI;break;
					default: res = atof(Temp2);
				}
			}
			else{
				
				// special functions 
				double arg;
				
				int len3 = strlen(Temp2);
				char* Temp3 = (char*)malloc(max(len3+1,2)*sizeof(char));
				if (Pos==Temp2){				
					region_strcpy(Temp3,&(Temp2[1]),&(Temp2[len3-1]));
					res = parse_expression2D(Temp3,x,y,t);
				}
				else{
					region_strcpy(Temp3,&(Temp2[0]),Pos);
					char* Temp4 = (char*)malloc(max(len3+1,2)*sizeof(char));
					region_strcpy(Temp4,Pos+1,&(Temp2[len3-1]));
					arg = parse_expression2D(Temp4,x,y,t);
					if (strcmp(Temp3,"sin")==0) res = sin(arg);
					else if (strcmp(Temp3,"cos")==0) res = cos(arg);
					else if (strcmp(Temp3,"exp")==0) res = exp(arg);
					else if (strcmp(Temp3,"tan")==0) res = tan(arg);
					else if (strcmp(Temp3,"atan")==0) res = atan(arg);
					else if (strcmp(Temp3,"cosh")==0) res = cosh(arg);
					else if (strcmp(Temp3,"sinh")==0) res = sinh(arg);
					else if (strcmp(Temp3,"tanh")==0) res = tanh(arg);
					else if (strcmp(Temp3,"atanh")==0) res = atanh(arg);
					else if (strcmp(Temp3,"log")==0) res = log(arg);
					else if (strcmp(Temp3,"sqrt")==0) res = sqrt(arg);
					else if (strcmp(Temp3,"theta")==0) res = (arg>=0 ? 1. : 0);
					else if (strcmp(Temp3,"stheta")==0) res = (tanh(150.*arg)+1.)/2.;
					free(Temp4);
				}
				free(Temp3);		
			}
		}
		free(Temp2);
	}
	free(Temp);
	//printf("expr: %s -> evaluated: %f\n",Expression,res);
	return res;
}

double parse_bound_expr(char* Expr,double x,double y,double t){
	
	#define GET_PART {\
				Part = strtok(NULL,","); \
				if (Part==NULL) return NAN; \
	}
	
	char* Part;
	char Temp[512];
	strcpy(Temp,Expr);
	if (outer_brackets(Temp)) remove_brackets(Temp);
	
	double res = 0;
	Part = strtok(Temp,",");
	if (strcmp(Part,CONST)==0){
		GET_PART;
		return atof(Part);
	}
	else if (strcmp(Part,LINEAR)==0){
		GET_PART;
		double period = atof(Part);
		GET_PART;
		double min = atof(Part);
		GET_PART;
		double max = atof(Part);
		return linear(t,period,min,max);
	}
	else if (strcmp(Part,LIN_SATURATION)==0){
		GET_PART;
		double c = atof(Part);
		GET_PART;
		double min = atof(Part);
		GET_PART;
		double max = atof(Part);
		return linear_saturation(t,c,min,max);
	}
	else if (strcmp(Part,TRIANG)==0){
		GET_PART;
		double period = atof(Part);
		GET_PART;
		double offset = atof(Part);
		GET_PART;
		double min = atof(Part);
		GET_PART;
		double max = atof(Part);
		return linear_updown_signal(t,period,offset,min,max);
	}
	else if (strcmp(Part,SAW)==0){
		GET_PART;
		double period = atof(Part);
		GET_PART;
		double offset = atof(Part);
		GET_PART;
		double min = atof(Part);
		GET_PART;
		double max = atof(Part);
		return saw_signal(t,period,offset,min,max);
	}
	else if (strcmp(Part,RECT)==0){
		GET_PART;
		double period = atof(Part);
		GET_PART;
		double offset = atof(Part);
		GET_PART;
		double min = atof(Part);
		GET_PART;
		double max = atof(Part);
		return rect_signal(t,period,offset,min,max);				
	}
	else if (strcmp(Part,RAMP)==0){
		GET_PART;
		double period = atof(Part);
		GET_PART;
		double offset = atof(Part);
		GET_PART;
		double min = atof(Part);
		GET_PART;
		double max = atof(Part);
		return ramp_signal(t,period,offset,min,max);	
	}
	else if (strcmp(Part,STEP)==0){
		GET_PART;
		double t_c = atof(Part);
		GET_PART;
		double dt = atof(Part);
		GET_PART;
		double v1 = atof(Part);
		GET_PART;
		double v2 = atof(Part);
		return step_signal(t,t_c,dt,v1,v2);	
	}
	return 0;
	
	#undef GET_PART
}

void free_bound_info(bound_info** Info){
	int i,j;
	for (i=0;i<(*Info)->loops;i++){
		for (j=0;j<(*Info)->Sizes[i];j++) free((*Info)->Time_depend[i][j]);
		free((*Info)->Coords[i]);
		free((*Info)->Cond[i]);	
	}
	free((*Info)->Sizes);
	free(*Info);
}

void free_bound_cond(bound_cond** Cond){
	free((*Cond)->Cond);
	free((*Cond)->Val);
	free(*Cond);
}
