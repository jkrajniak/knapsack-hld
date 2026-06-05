
/* ======================================================================
	      MCKNAP.c, David Pisinger   oct 1993, modified july 1994
   ====================================================================== */

/* This is the C-code corresponding to the paper:
 *
 *   D. Pisinger
 *   A minimal algorithm for the multiple-choice knapsack problem
 *   European Journal of Operational Research, 83 (1995), 394-410
 *
 * Further details on the project can also be found in
 *
 *   D. Pisinger
 *   Algorithms for Knapsack Problems
 *   Report 95/1, DIKU, University of Copenhagen
 *   Universitetsparken 1
 *   DK-2100 Copenhagen
 *
 * The current code is intended for performing extensive tests with
 * randomly generated instances. It should however be easy to derive
 * the "plain" mcknap algorithm from the listing by stripping several
 * test routines.
 *
 * The code has been tested on a hp9000/735, and conforms with the
 * ANSI-C standard apart from some of the timing routines (which may
 * be removed). To compile the code use:
 *
 *   cc -Aa -O -o mcknap mcknap.c -lm
 *
 * The code is run by issuing the command
 *
 *   mcknap k n r type
 *
 * where k: number of classes,
 *       n: number of items in each class,
 *       r: range of coefficients,
 *       type: 1=uncorr., 2=weakly corr., 3=strongly corr., 4=subset sum
 *             5=zig-zag (sinha-zoltners), 6=0-1knapsack problem
 * output will be appended to the file "trace.mc".
 *
 * The code may only be used for academic or non-commercial purposes.
 * Errors and questions are refered to:
 *
 *   David Pisinger, associate professor
 *   DIKU, University of Copenhagen,
 *   Universitetsparken 1,
 *   DK-2100 Copenhagen.
 *   e-mail: pisinger@diku.dk
 *   fax: +45 35 32 14 01
 */


/* ======================================================================
                                  definitions
   ====================================================================== */

#define TRACELEVEL  0                /* level of debug information */
#define START       1                /* first test to be run */
#define TESTS       100              /* last test to be run */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <values.h>
#include <math.h>
#include <string.h>
#include <values.h>
#include <limits.h>
#include <malloc.h>
#define _INCLUDE_POSIX_SOURCE
#include <sys/times.h>
#include <unistd.h>


/* ======================================================================
				   macros
   ====================================================================== */

#define srand(x)    srand48(x)
#define random(x)   (lrand48() % (x))

#define SYNC           5   /* when to switch to linear scan in binary scan */
#define MEDIMAX        15
#define MAXSTACK       100
#define MAXLIST        32
#define MAXVTYPE       ULONG_MAX

#define TRUE           1
#define FALSE          0

#define MAXIMIZE       0
#define MINIMIZE       1

#define DET(a1, a2, b1, b2)    ((a1) * (stype) (b2) - (a2) * (stype) (b1))
#define SWAPS(a,b)      { register itemset t; t=*(a); *(a)=*(b); *(b)=t; }
#define SWAPI(a,b)      { register itemrec t; t=*(a); *(a)=*(b); *(b)=t; }
#define SWAPO(a,b)      { register ordrec  t; t=*(a); *(a)=*(b); *(b)=t; }
#define SIZE(a)                          ((int) (((a)->lset+1)-(a)->fset))


/* ======================================================================
				 type declarations
   ====================================================================== */

typedef int           boolean; /* logical variable */
typedef int           ntype;   /* number of stages */
typedef short         itype;   /* item profits and weights */
typedef long          stype;   /* sum of pofit or weight */
typedef unsigned long vtype;   /* solution vector */

/* partial vector */
typedef struct {
  stype    psum;
  stype    wsum;
  vtype    vect;
} partvect;

/* item */
typedef struct {
  itype    psum;
  itype    wsum;
} itemrec;

/* set of partial vectors */
typedef struct {
  ntype    size;
  itemrec  *fset;
  itemrec  *lset;
  itemrec  *no;
  itemrec  f,l;
  boolean  used;
} itemset;

/* set of partial vectors */
typedef struct {
  ntype    size;
  partvect *fset;
  partvect *lset;
} partset;

/* set of itemsets */
typedef struct {
  itemset  *fset;
  itemset  *lset;
  ntype    size;
} isetset;

/* order record */
typedef struct {
  itype    dp;
  itype    dw;
  itemset  *ref;
} ordrec;

/* order interval */
typedef struct {
  ordrec   *f;
  ordrec   *l;
} ordintv;

/* order stack */
typedef struct {
  ordintv  intv[MAXSTACK];
  int      level;
  int      optim;
  ordrec   *first;
  ordrec   *last;
  ordrec   *i;
} ordstack;

/* solution record */
typedef struct {
  ntype    size;
  itemset  *set;
} solrec;

/* solution structure */
typedef struct {
  solrec   list[MAXLIST];
  ntype    size;
  stype    psum;
  stype    wsum;
  vtype    vect;
  vtype    vmax;
  ordrec   *a;
  ordrec   *b;
} solstruct;

typedef int (*funcptr) (const void *, const void *);


typedef struct { /* all problem information */
  ntype k;
  ntype n;
  int   type;
  itype range;

  stype capacity;	      /* capacity of knapsack */
  stype dantzig;              /* the dantzig upper bound */
  stype zstar;                /* optimal solution */
  stype summul;	  	      /* sum of multiplications */
  stype antmul;		      /* number of multiplications */
  stype maxmul;               /* max multiplied set */
  stype redusets;             /* sum of reduced sets */
  stype reduitems;            /* sum of items which are tested for reduce */
  stype redukill;             /* sum of tested items which were reduced */
  stype gap;                  /* current gap */
  stype partitions;           /* number of partitions */
  stype domikill;             /* number of dominated-kills */
  stype lpkill;               /* number of lp-kills */
  long  timepar;              /* time used for partitioning */
  long  timesort;             /* time used for sorting of gradients */
  long  time;                 /* time used for all solution */
  long  welldef;              /* is the found solution correct */
  long  checked;              /* optimal solution checked */
  long  iterates;             /* number of iterations to find optimal sol */
} allinfo;


/* ======================================================================
				  global variables
   ====================================================================== */

solstruct solution;
solstruct optsol;


/* ======================================================================
				   debug
   ====================================================================== */

FILE *trace;
int traceniveau;
int tracelevel;
int traceconsole;
double tot1, tot2;
char lines[100];

/* due to failure in include library, the following must be included */
struct tms timestart, timeend;


void starttime(void)
{
  times(&timestart);
}


void showtime(void)
{
  double t1, t2;
  times(&timeend);
  t1 = (double) (timeend.tms_utime-timestart.tms_utime) / sysconf(_SC_CLK_TCK);
  fprintf(trace, "%.2lf", t1);
}


void endtime(long *time)
{
  double t1, t2;
  times(&timeend);
  t1 = (double) (timeend.tms_utime-timestart.tms_utime) / sysconf(_SC_CLK_TCK);
  t2 = (double) (timeend.tms_stime-timestart.tms_stime) / sysconf(_SC_CLK_TCK);
  tot1 += t1; tot2 += t2;
  *time = t1 * 1000;
}


void inittrace(char *ext)
{
  int i;
  char s[100];

  strcpy(s, "trace.");
  strcat(s, ext);
  trace = fopen(s,"a");
  if (trace == NULL) printf("trace not openn\n");
  tot1 = 0; 
  tot2 = 0;
  tracelevel = TRACELEVEL;
  traceniveau = 0;
  traceconsole = 0;
  if (tracelevel < 0) { traceconsole = 1; tracelevel = -tracelevel; }
  for (i = 0; i < 100; i+=2) { lines[i] = '|'; lines[i+1] = ' '; }
}


void inds(int level, char *s, ...)
{
  va_list args;
  char t[500];

  if (level <= tracelevel) {
    va_start(args, s);
    memcpy(t, lines, 2*traceniveau);
    strcpy(t + 2*traceniveau, "/");
    vsprintf(t + 2*traceniveau + 1, s, args);
    fprintf(trace, "%s\n", t);
    if (traceconsole) printf("%s\n", t);
    va_end(args);
  }
  traceniveau++;
}


void vis(int level, char *s, ...)
{
  va_list args;
  char t[500];

  if (level <= tracelevel) {
    va_start(args, s);
    memcpy(t, lines, 2*traceniveau);
    vsprintf(t + 2*traceniveau, s, args);
    if (level == 0) {
      printf("%s", t);
      fprintf(trace, "%s", t);
    } else {
      fprintf(trace, "%s", t);
      if (traceconsole) printf("%s", t);
    }
    va_end(args);
  }
}


void uds(int level, char *s, ...)
{
  va_list args;
  char t[500];

  traceniveau--;
  if (level <= tracelevel) {
    va_start(args, s);
    memcpy(t, lines, 2*traceniveau);
    strcpy(t + 2*traceniveau, "\\");
    vsprintf(t + 2*traceniveau + 1, s, args);
    fprintf(trace, "%s\n", t);
    if (traceconsole) printf("%s\n", t);
    va_end(args);
  }
}


void closetrace(void)
{
  int i;

  fflush(trace); 
  fclose(trace);
}


/* =======================================================================
				  error
   ======================================================================= */

void error(char *str, ...)
{
  va_list args;

  va_start(args, str);
  vprintf(str, args);
  printf("\n");
  vfprintf(trace, str, args);
  fprintf(trace, "\n");
  va_end(args);
  printf("THE PROGRAM IS TERMINATED !!!\n\n");
  fprintf(trace, "THE PROGRAM IS TERMINATED !!!\n\n");
  closetrace();
  exit(-1);
}


/* ======================================================================
                                  checktime
   ====================================================================== */

void checktime(void)
{
  boolean first;
  struct tm *t;
  time_t s;

  for (first = TRUE;; first = FALSE) {
    break;
    s = time(NULL);
    t = localtime(&s);
    if ((t->tm_wday == 0) || (t->tm_wday ==  6)) break;
    if ((t->tm_hour <  8) || (t->tm_hour >= 17)) break;
    if (first) printf("sleeping\n");
    sleep(60);
  }
}


/* ======================================================================
				  palloc
   ====================================================================== */

void * palloc(long size)
{
  void * m;
  if (size == 0) size = 1;
  if (size != (size_t) size) error("Alloc too big %ld", size);
  m = malloc(size);
  if (m == NULL) error("Alloc no space %ld", size);
  return m;
}


void pfree(void *p)
{
  if (p == NULL) error("Freeing NULL");
  free(p);
}


/* ======================================================================
				  freeitems
   ====================================================================== */

void freeitems(isetset *h)
{
  itemset *im, *i;

  /* free current partial vectors */
  im = h->lset;
  for (i = h->fset; i <= im; i++) {
    pfree(i->fset);
  }

  /* free hset */
  pfree(h->fset);
}


/* ======================================================================
				  pushstack
   ====================================================================== */

void pushstack(ordstack *stack, ordrec *f, ordrec *l)
{
  int v;

  (stack->level)++; v = stack->level;
  if (v == MAXSTACK) error("stack filled");
  stack->intv[v].f = f;
  stack->intv[v].l = l;
}


/* ======================================================================
				 vispartset
   ====================================================================== */

void vispartset(partset *d)
{
  partvect *i, *im;

  vis(1,"SET: size %d\n", (int) d->size);
  im = d->lset + 1;
  for (i = d->fset; i < im; i++) {
    vis(1,"    (%ld,%ld)\n", i->psum, i->wsum);
  }
}


/* ======================================================================
				 visitems
   ====================================================================== */

void visitems(itemset *d)
{
  itemrec *i, *im;

  vis(1,"ITEMSET: size %d choice (%hd,%hd)\n",
      d->size, d->f.psum, d->f.wsum);
  im = d->lset;
  for (i = d->fset; i <= im; i++) {
    vis(1,"    (%hd,%hd)\n", i->psum, i->wsum);
  }
}


/* ======================================================================
			       sumdata
   ====================================================================== */

void sumdata(allinfo *a)
{
  static long k;
  static long n;
  static long r;
  static long t;
  static long summul      = 0;
  static long antmul      = 0;
  static long maxmul      = 0;
  static long redusets    = 0;
  static long reduitems   = 0;
  static long redukill    = 0;
  static long gap         = 0;
  static long partitions  = 0;
  static long domikill    = 0;
  static long lpkill      = 0;
  static long timepar     = 0;
  static long timesort    = 0;
  static long tottime     = 0;
  static long welldef     = 0;
  static long checked     = 0;
  static long iterates    = 0;
  static long zsum        = 0;
  static long csum        = 0;
  static double sqtime    = 0.0;
  double mean, variance, stddev, redupct;

  if (a == NULL) {
    mean     = tottime / (1000 * (double) TESTS);
    variance = sqtime / TESTS - mean * mean;
    stddev   = sqrt(variance);
    redupct  = (reduitems == 0 ? 
                0.0 : 100 * (double) redukill / (double) reduitems);
    fprintf(trace,"k        = %ld\n", k);
    fprintf(trace,"n        = %ld\n", n);
    fprintf(trace,"r        = %ld\n", r);
    fprintf(trace,"t        = %ld\n", t);
    fprintf(trace,"summul   = %.0lf\n", summul         / (double)     TESTS);
    fprintf(trace,"antmul   = %.0lf\n", antmul         / (double)     TESTS);
    fprintf(trace,"maxmul   = %.0lf\n", .001*maxmul    / (double)         1);
    fprintf(trace,"rclass   = %.0lf\n", 100*redusets   / (double) (k*TESTS));
    fprintf(trace,"ritems   = %.0lf\n", reduitems      / (double)     TESTS);
    fprintf(trace,"rkill    = %.0lf\n", redukill       / (double)     TESTS);
    fprintf(trace,"rpct     = %.0lf\n", redupct        / (double)         1);
    fprintf(trace,"gap      = %.1lf\n", gap            / (double)     TESTS);
    fprintf(trace,"part     = %.1lf\n", partitions     / (double)     TESTS);
    fprintf(trace,"domikill = %.0lf\n", domikill       / (double)     TESTS);
    fprintf(trace,"lpkill   = %.0lf\n", lpkill         / (double)     TESTS);
    fprintf(trace,"zsum     = %.0lf\n", zsum           / (double)         1);
    fprintf(trace,"csum     = %.0lf\n", csum           / (double)         1);
    fprintf(trace,"timepar  = %.2lf\n", 0.001*timepar  / (double)     TESTS);
    fprintf(trace,"timesort = %.2lf\n", 0.001*timesort / (double)     TESTS);
    fprintf(trace,"tottime  = %.2lf\n", 0.001*tottime  / (double)     TESTS);
    fprintf(trace,"variance = %.2lf\n", variance       / (double)         1);
    fprintf(trace,"stddev   = %.2lf\n", stddev         / (double)         1);
  } else {
    summul      += a->summul / 1000;
    antmul      += a->antmul;
    if (a->maxmul > maxmul) maxmul = a->maxmul;
    redusets    += a->redusets;
    reduitems   += a->reduitems;
    redukill    += a->redukill;
    gap         += a->gap;
    partitions  += a->partitions;
    domikill    += a->domikill;
    lpkill      += a->lpkill;
    zsum         = ((zsum + a->zstar   ) % 1000);
    csum         = ((csum + a->capacity) % 1000);
    welldef     += a->welldef;
    checked     += a->checked;
    iterates    += a->iterates;
    timepar     += a->timepar;
    timesort    += a->timesort - a->timepar;
    tottime     += a->time;
    sqtime      += (a->time / (double) 1000) * (a->time / (double) 1000);

    k = a->k;
    n = a->n;
    r = a->range;
    t = a->type;
  }
}


/* ======================================================================
				 checksolution
   ====================================================================== */

void checksolution(allinfo *a, isetset *head, stype zstar, stype cstar)
{
  itemset *jm, *j;
  stype psum, wsum;

  psum = wsum = 0;
  jm = head->lset;
  for (j = head->fset; j <= jm; j++) {
    psum += j->f.psum;
    wsum += j->f.wsum;
  }
  vis(1,"CHECKSOLUTION %ld,%ld TO %ld,%ld DANTZIG %ld\n",
	 psum, wsum, zstar, cstar, a->dantzig);
  a->checked = (psum == zstar) && (wsum <= cstar) && (zstar <= a->dantzig);
  if (!a->checked) {
    vis(0,"WRONG CHECKSOLUTION %ld,%ld TO %ld,%ld DANTZIG %ld\n",
	   psum, wsum, zstar, cstar, a->dantzig);
  }
}


/* ======================================================================
				  rotatesol
   ====================================================================== */

void rotatesol(partset *a, itemset *b)
{
  register partvect *j;
  partvect *jm;
  ntype size, i;
  vtype vmax;

  /* check for sufficiency */
  size = b->size;
  while (MAXVTYPE / solution.vmax < size) {
    solution.vmax = solution.vmax / solution.list[solution.size-1].size;
    solution.size--;
  }
  vmax = solution.vmax;

  /* rotate array left */
  solution.size++;
  for (i = solution.size; i >= 1; i--) {
    solution.list[i] = solution.list[i-1];
  }

  /* place at end */
  solution.list[0].size = size;
  solution.list[0].set  = b;
  solution.vmax = vmax * size;

  /* now rotate partset */
  jm = a->lset;
  for (j = a->fset; j <= jm; j++) j->vect = (j->vect % vmax) * size;
}


/* ======================================================================
				  savesol
   ====================================================================== */

void savesol(partvect *v, ordrec *a, ordrec *b)
{
  optsol = solution;
  optsol.psum = v->psum;
  optsol.wsum = v->wsum;
  optsol.vect = v->vect;
  optsol.a    = a;
  optsol.b    = b;
}


/* ======================================================================
				  definesol
   ====================================================================== */

boolean definesol(allinfo *al, stype fixp, stype fixw,
                  ordstack *a, ordstack *b, stype *c, stype *z, stype *ub)
{
  vtype vect, rem;
  stype psum, wsum;
  itemrec *jc;
  ordrec *i;
  solrec *s;
  int k;

  vect = optsol.vect;
  psum = optsol.psum; wsum  = optsol.wsum;
  vis(1,"definesolution size %d, (%ld,%ld) vect %ld vmax %ld\n",
      optsol.size, psum, wsum, vect, optsol.vmax);
  vis(1,"definesolution maengde a: %d, b: %d\n",
     optsol.a - a->first, b->last - optsol.b);

  /* prepare sets for next iteration */
  for (i = a->first; i <= optsol.a; i++)
    if (i->ref != NULL) i->ref->used = FALSE;
  for (i = optsol.b; i <= b->last;  i++)
    if (i->ref != NULL) i->ref->used = FALSE;

  /* find solution vector */
  for (k = 0; k < optsol.size; k++) {
    s = &(optsol.list[k]);
    rem  = vect  % s->size;
    jc = s->set->fset + rem;
    vis(2,"choice no %ld of %hd is (%hd,%hd)\n",
	rem, s->size, jc->psum, jc->wsum);
    psum -= jc->psum - s->set->f.psum;
    wsum -= jc->wsum - s->set->f.wsum;
    s->set->f    = *jc;  /* save choice in f */
    s->set->used = TRUE; /* avoid recalculation */
    vect  = vect  / s->size;
  }
  vis(2,"FINAL sum %ld,%ld, break %ld,%ld\n", psum, wsum, fixp, fixw);
  al->welldef = (fixp == psum) && (fixw == wsum);
  if (al->welldef) return TRUE;

  /* new problem */
  *z = psum - 1; *c = wsum; *ub = psum;
  a->i = a->first; b->i = b->last;
  return FALSE;
}


/* ======================================================================
				  findvect
   ====================================================================== */

partvect * findvect(stype ws, partvect *f, partvect *l)
{
  /* find vector i, so that i->wsum <= ws < (i+1)->wsum */
  partvect *m;

  /* a set should always have at least one vector */
  if (f > l) error("findvect: empty set");

  if (f->wsum >  ws) error("findvect: too big");
  if (l->wsum <= ws) return l;

  while (l - f > SYNC) {
    m = f + (l - f) / 2;
    if (m->wsum > ws) {
      l = m-1;
    } else {
      f = m;
    }
  }
  while (l->wsum > ws) l--;

  if (l->wsum     >  ws) error("findvect: too big l");
  if ((l+1)->wsum <= ws) error("findvect: too small l");
  return l;
}


/* ======================================================================
				  maketest
   ====================================================================== */

int icmp(itype *a, itype *b) { return *a - *b; }

void maketest(itemset *j, int r, int type)
{
  register itemrec *i;
  itemrec *im;
  itype *p, *w;
  stype ps, ws;
  ntype k, n;

  n = j->size;
  if (type == 3) r = (2*r) / n;
  if ((type == 3) || (type == 5)) {
    p = palloc(n * (long) sizeof(itype));
    w = palloc(n * (long) sizeof(itype));
    for (k = 0; k < n; k++) {
      w[k] = random(r) + 1;
      p[k] = (type == 3 ? w[k] + 10 : random(r) + 1);
    }
    qsort(p, n, sizeof(itype), (funcptr) icmp);
    qsort(w, n, sizeof(itype), (funcptr) icmp);
  }

  im = j->lset; ps = 0; ws = 0;
  for (i = j->fset; i <= im; i++) {
    switch (type) {
      case  1: i->wsum = random(r) + 1;
	       i->psum = random(r) + 1;
	       break;
      case  2: i->wsum = random(r) + 1;
	       i->psum = random(21) + i->wsum - 10;
	       if (i->psum <= 0) i->psum = 1;
	       break;
      case  3: i->wsum = w[i - j->fset] + ws;
	       i->psum = p[i - j->fset] + ps;
	       break;
      case  4: i->wsum = random(r) + 1;
	       i->psum = i->wsum;
	       break;
      case  5: i->wsum = w[i - j->fset];
	       i->psum = p[i - j->fset];
	       break;
      case  6: if (i == j->fset) { i->wsum = 0; i->psum = 0; break; }
	       i->wsum = random(r) + 1;
	       i->psum = random(r) + 1;
	       break;
      default: i->wsum = 0;
	       i->psum = 0;
	       break;
    }
    ws = i->wsum; ps = i->psum;
  }

  if ((type == 3) || (type == 5)) {
    pfree(p);
    pfree(w);
  }
}


/* ======================================================================
				 inititems
   ====================================================================== */

stype inititems(allinfo *al, isetset *h,
                ntype classes, int size, int r, int type)
{
  itemset *j, *jm, *k;
  itemrec *i, *im, *mi, *ma;
  stype wsum1, wsum2;

  /* init itemset */
  h->size  = classes;
  h->fset  = palloc(h->size * (long) sizeof(itemset));
  h->lset  = h->fset + h->size - 1;

  /* generate test classes */
  jm = h->lset;
  for (j = h->fset; j <= jm; j++) {
    j->size = size;
    j->fset = palloc(size * (long) sizeof(itemrec));
    j->lset = j->fset + size - 1;
    maketest(j, r, type);
  }

  /* find c as 1/2 of extreme weights in each set */
  wsum1 = 0; wsum2 = 0;
  for (j = h->fset; j <= jm; j++) {
    im = j->lset;
    mi = im; ma = im;
    for (i = j->fset; i < im; i++) {
      if (i->wsum < mi->wsum) mi = i;
      if (i->psum >= ma->psum) {
	if ((i->psum > ma->psum) || (i->wsum < ma->wsum)) ma = i;
      }
    }
    wsum1 += mi->wsum;
    wsum2 += ma->wsum;
  }
  al->capacity = (wsum1 + wsum2) / 2;
  vis(2,"SETS %hd, MINW %ld, MAXW %ld, C %ld\n",
      classes, wsum1, wsum2, al->capacity);
  return al->capacity;
}


/* ======================================================================
				    merge
   ====================================================================== */

void merge(partset *jset, itemset  *iset, ntype f, ntype l,
	   partvect **k1, partvect **km)
{
  if (f == l) {
    register partvect *j, *k;
    register itype psum, wsum;
    partvect *j1, *jm;
    itemrec *i;
    ntype d;

    d  = jset->size;
    j1 = jset->fset; jm = jset->lset;
    *k1 = palloc((d+1) * (long) sizeof(partvect));/* 1 extra is used below */
    *km = *k1 + d - 1;
    i = iset->fset + f;         /* add item i minus lp-choice in set iset */
    psum = i->psum - iset->f.psum;
    wsum = i->wsum - iset->f.wsum;
    for (k = *k1, j = j1; j <= jm; k++, j++) {
      k->psum = j->psum + psum;
      k->wsum = j->wsum + wsum;
      k->vect = j->vect + f;
    }
  } else {
    register partvect *k, *a, *b;
    partvect *a1, *am, *b1, *bm;
    long size;
    ntype d;

    d = (l - f) / 2;
    merge(jset, iset, f,     f+d, &a1, &am);
    merge(jset, iset, f+d+1, l,   &b1, &bm);

    size = (am - a1 + 1) + (long) (bm - b1 + 1) + 1; /* 1 extra used below */
    *k1 = palloc(size * (long) sizeof(partvect));
    a = a1; b = b1; k = *k1;
    if (a->wsum <= b->wsum) { *k = *a; a++; } else { *k = *b; b++; }
    (am+1)->wsum = bm->wsum + 1; (am+1)->psum = 0;/* add max as extra item */
    (bm+1)->wsum = am->wsum + 1; (bm+1)->psum = 0;
    am++; bm++;

    for (;;) {
      if (a->wsum <= b->wsum) {
	if (a->psum > k->psum) {
	  if (a->wsum > k->wsum) k++;
	  *k = *a;
	} a++;
      } else {
	if (b->psum > k->psum) {
	  if (b->wsum > k->wsum) k++;
	  *k = *b;
	} b++;
      }
      if ((a == am) && (b == bm)) break;
    }
    *km = k;
    pfree(a1);
    pfree(b1);
  }
}


/* ======================================================================
				  multiply
   ====================================================================== */

void multiply(allinfo *al, partset *a, itemset *b)
{
  partvect *k1, *km;
  ntype size;
  vtype vmax;

  rotatesol(a, b);
  merge(a, b, 0, b->size-1, &k1, &km);
  pfree(a->fset);
  a->fset = k1;
  a->lset = km;
  size = SIZE(a);
  vis(2,"MULTIPLY (%ld*%ld) = %ld -> %ld\n",
      (long) a->size, (long) b->size, a->size * (long) b->size, (long) size);
  a->size = size;
  if (size > al->maxmul) al->maxmul = size;
}


/* ======================================================================
				  reduceset
   ====================================================================== */

void reduceset(partset *a, ordrec *t, ordrec *s, stype *z1, stype c)
{
  register partvect *i, *k;
  register itype ps, ws, pt, wt;
  register stype z;
  partvect *r1, *rm, *v;

  if (a->size == 0) return;
  pt = t->dp; wt = t->dw;
  ps = s->dp; ws = s->dw;

  /* initialize limits */
  r1 = a->fset;
  rm = a->lset;

  if (r1->wsum > c) {
    /* alle overvaegtige */
    v = r1 - 1;
  } else {
    v = findvect(c, r1, rm);
    if (v->psum > *z1) {
      savesol(v, t, s);
      *z1 = v->psum;
    }
  }
  z = *z1 + 1;

  /* now do the reduction */
  k = r1;
  for (i = r1; i != v+1; i++) {
    if (DET(i->psum-z, i->wsum-c, pt, wt) >= 0) { *k = *i; k++; }
  }
  for (i = v+1; i <= rm; i++) {
    if (DET(i->psum-z, i->wsum-c, ps, ws) >= 0) { *k = *i; k++; }
  }
  vis(2,"Z=%ld, reduceset %3d -> %3d  s(%hd,%hd) t(%hd,%hd)\n",
      *z1, (int) a->size, (int) (k - a->fset), ps, ws, pt, wt);
  a->lset = k - 1;
  a->size = SIZE(a);
}


/* ======================================================================
				  reduceitem
   ====================================================================== */

void reduceitem(allinfo *al, itemset *a, stype psum, stype wsum,
		itype pb, itype wb, stype z1, stype c)
{
  register itemrec *i;
  stype z, ub, psum1, wsum1;
  itemrec *i1, *im;

  if (a->size == 1) return;

  /* sum of fixed items in greedy solution */
  psum -= a->f.psum;
  wsum -= a->f.wsum;

  al->redusets++;
  /* now do the reduction */
  i1 = a->fset; im = a->lset; z = z1 + 1;
  for (i = i1; i <= im; ) {
    al->reduitems++;
    psum1 = i->psum + psum; wsum1 = i->wsum + wsum;
    if (DET(psum1-z,wsum1-c,pb,wb) >= 0) { i++; } 
    else { SWAPI(i,im); im--; al->redukill++; }
  }
  vis(2,"Z=%ld, reitem %3d -> %3d\n",
      z1, (int) a->size, (int) (i - a->fset));
  a->lset = i - 1;
  a->size = SIZE(a);
}


/* ======================================================================
				  preprocess
   ====================================================================== */

void preprocess(isetset *head, stype *fixp, stype *fixw,
		 stype *minw1, stype *maxw1)
{
  register itemrec *i;
  itemrec *im, *f, *l;
  itemset *j, *jm;
  stype minw, maxw, lw;
  long kill, setout;

  vis(2,"\nPreprocess sets %hd ", head->size);
  minw = maxw = 0; kill = setout = 0;
  jm = head->lset;
  for (j = head->fset; j <= jm; ) {
    im = j->lset;
    f = l = im;
    for (i = j->fset; i < im; i++) {
      if (i->wsum <  f->wsum) f = i;
      if (i->psum >= l->psum) {
	if ((i->psum > l->psum) || (i->wsum < l->wsum)) l = i;
      }
    }
    j->f = *f; minw += f->wsum;
    j->l = *l; maxw += l->wsum;

    /* now remove dominated */
    lw = l->wsum; SWAPI(j->fset,l);
    for (i = j->fset+1; i <= im; ) {
      if (i->wsum >= lw) { SWAPI(i, im); im--; kill++; } else {	i++; }
    }
    j->lset = im;
    if (j->fset == j->lset) {
      *fixp += im->psum;
      *fixw += im->wsum;
      SWAPS(j, jm); jm--;
      setout++;
    } else {
      j++;
    }
  }
  head->lset = jm;
  vis(2,"now %hd fix (%ld,%ld) kill %hd\n", head->size, *fixp, *fixw, kill);
  *minw1 = minw; *maxw1 = maxw;
}


/* ======================================================================
				  choosemedian
   ====================================================================== */

int lamless(ordrec *a, ordrec *b)
{
  register stype sum;
  sum = DET(a->dp, a->dw, b->dp, b->dw);
  if (sum < 0) return -1;
  return (sum > 0);
}


void choosemedian(isetset *head, itype *cdp, itype *cdw)
{
  int d;
  itype dp, dw;
  itemset *i;
  ordrec a[MEDIMAX];

  for (d = 0; d < MEDIMAX; d++) {
    i = head->fset + random(SIZE(head)); /* random choice */
    a[d].dp = i->l.psum - i->f.psum;
    a[d].dw = i->l.wsum - i->f.wsum;
  }

  qsort(&a, d, sizeof(ordrec), (funcptr) lamless);
  *cdp = a[d/2].dp; *cdw = a[d/2].dw;
  vis(2,"\nmedian (%hd,%hd)\n", *cdp, *cdw);
}


/* ======================================================================
				  outermost
   ====================================================================== */

void outermost(isetset *head, itype dp, itype dw,
	       stype *minwsum, stype *maxwsum)
{
  register itemrec *i, *no;
  itemrec *i1, *im;
  itemset *j, *jm;
  stype sum, msum;

  jm = head->lset;
  for (j = head->fset; j <= jm; j++) {

    /* find outermost item */
    i1 = j->fset; im = j->lset; msum = LONG_MIN;
    for (i = i1; i <= im; i++) {
      sum = DET(i->psum, i->wsum, dp, dw);
      if (sum >= msum) {
	if (sum > msum) { msum = sum; no = i1; }
	SWAPI(no, i); no++;
      }
    }
    no--;

    /* determine min and max weightsums */
    if (no != i1) {
      if (i1->wsum > no->wsum) SWAPI(i1, no);
      for (i = i1+1; i < no; i++) {
	if (i->wsum < i1->wsum) { SWAPI(i, i1); } else
	if (i->wsum > no->wsum) { SWAPI(i, no); }
      } /* now i1->wsum minimal, no->wsum maximal */
    }
    *minwsum += i1->wsum;
    *maxwsum += no->wsum;
    j->no = no;
  }
}


/* ======================================================================
				  separete
   ====================================================================== */

void separate(allinfo *al, isetset *head, boolean underfull,
              stype *fixp, stype *fixw)
{
  register itemrec *i, *im;
  itemrec *i1;
  itemset *j, *jm;
  itype p1, w1, pm, wm;
  long dkill, left, setout;

  dkill = left = setout = 0;
  jm = head->lset;
  for (j = head->fset; j <= jm; ) {

    /* choose min and max items for partitioning */
    i1 = j->fset; im = j->lset;
    if (underfull) { SWAPI(j->no, i1); j->f = *i1; } else { j->l = *i1; }
    p1 = j->f.psum; wm = j->l.wsum;

    /* delete dominated or too big/small items */
    if (underfull) {
      for (i = i1+1; i <= im; ) {
	if (i->psum <= p1) { SWAPI(i, im); im--; dkill++; }
	else { i++; left++; }
      }
    } else {
      for (i = i1+1; i <= im; ) {
	if (i->wsum >= wm) { SWAPI(i, im); im--; dkill++; }
	else { i++; left++; }
      }
    }
    j->lset = im;

    /* check for singleton sets */
    if (j->fset == j->lset) {
      *fixp += j->fset->psum;
      *fixw += j->fset->wsum;
      SWAPS(j, jm); jm--;
      setout++;
    } else {
      j++;
    }
  }
  head->lset    = jm;
  al->domikill += dkill;
  al->lpkill   += left;
  if (dkill == 0) error("partitioning with no domi-effect");
}


/* ======================================================================
				  optimum
   ====================================================================== */

void optimum(isetset *head, stype *fixp, stype *fixw, stype c)
{
  register itemrec *i;
  stype ps, ws;
  itemrec *i1, *im, *choice;
  itemset *j, *jm, *cut;

  /* define solution: first choose smallest items */
  jm = head->lset;
  for (j = head->fset; j <= jm; j++) {
    i1 = j->fset; *fixp += i1->psum; *fixw += i1->wsum;
  }

  /* then improve till filled */
  cut = NULL;
  for (j = head->fset; j <= jm; j++) {
    if ((*fixw == c) && (cut != NULL)) break;
    i1 = j->fset; im = j->no;
    for (i = i1+1; i <= im; i++) {
      ps = *fixp + i->psum - i1->psum;
      ws = *fixw + i->wsum - i1->wsum;
      if (ws <= c) {
	if (ps > *fixp) { *fixp = ps; *fixw = ws; SWAPI(i1, i); }
      } else {
	cut = j;
      }
    }
  }

  /* a set containing fractional variables is placed first */
  if (cut != NULL) SWAPS(head->fset, cut);
}


/* ======================================================================
				  partition
   ====================================================================== */

void partition(allinfo *al, isetset *head, stype c,
               stype *psum, stype *wsum, itype *dp, itype *dw)
{
  stype fixp, fixw, minwsum, maxwsum;

  fixp = 0; fixw = 0; al->partitions = 0;

  /* check for trivial solutions and reduce trivially dominated */
  preprocess(head, &fixp, &fixw, &minwsum, &maxwsum);
  if ((minwsum > c) || (maxwsum <= c)) return;

  for (;;) {
    al->partitions++;
    choosemedian(head, dp, dw);

    /* find projections in direction (dp,dw) */
    minwsum = fixw; maxwsum = fixw;
    outermost(head, *dp, *dw, &minwsum, &maxwsum);

    /* now consider the weight sums */
    vis(2,"wsum %ld-%ld: c %ld\n", minwsum, maxwsum, c);
    if ((minwsum <= c) && (c <= maxwsum)) break;

    /* separete set in dominated and live items */
    separate(al, head, (maxwsum < c), &fixp, &fixw);
  }
  /* now find the optimal lp-solution */
  optimum(head, &fixp, &fixw, c);

  *psum = fixp; *wsum = fixw;
  al->dantzig = fixp + ((c - fixw) * *dp) / *dw;
  vis(1,"UB=%ld Z=%ld FIXED (%ld,%ld) BREAK (%hd,%hd) CAP %ld\n",
      al->dantzig, fixp, fixp, fixw, *dp, *dw, al->capacity);
}


/* ======================================================================
				  restore
   ====================================================================== */

void restore(isetset *head, stype psum, stype wsum)
{
  register itemset *j, *jm;

  head->lset = head->fset + head->size - 1;
  jm = head->lset;
  for (j = head->fset; j <= jm; j++) {
    j->lset = j->fset + j->size - 1;
    j->used = FALSE;
    j->f    = *(j->fset);
    psum -= j->f.psum;
    wsum -= j->f.wsum;
  }
  if ((psum != 0) || (wsum != 0)) error("choices not first");
}


/* ======================================================================
				 domiitem
   ====================================================================== */

int itemless(itemrec *a, itemrec *b)
{
  itype sum;
  sum = a->wsum - b->wsum;
  if (sum != 0) return sum;
  return (a->psum - b->psum);
}


void domiitem(itemset *mid)
{
  register itemrec *i, *j, *k;
  itemrec *k1, *im, *i1;

  i1 = mid->fset; im = mid->lset;
  if (i1 == im) return;

  qsort(mid->fset, SIZE(mid), sizeof(itemrec), (funcptr) itemless);

  /* now remove dominated */
  k1 = palloc(mid->size * (long) sizeof(itemrec));
  for (i = i1+1, j = i1, k = k1; i <= im; i++ ) {
    if (i->psum > j->psum) { j++; *j = *i; } else { *k = *i; k++; }
  }
  mid->lset = j;
  mid->size = SIZE(mid);

  /* copy dominated to end of set */
  for (i = k1; i < k; i++) { j++; *j = *i; }
  pfree(k1);
  vis(2,"domiitem %hd\n", mid->size);
}


/* ======================================================================
				 initfirst
   ====================================================================== */

void initfirst(partset *mid, itemset *old, stype psum, stype wsum)
{
  itemrec *i, *i1, *im;
  partvect *j;

  domiitem(old);
  mid->fset = palloc(old->size * (long) sizeof(partvect));
  i1 = old->fset; im = old->lset;
  psum -= old->f.psum; wsum -= old->f.wsum; /* subtract lp-choice */
  for (i = i1, j = mid->fset; i <= im; i++) {
    j->psum = i->psum + psum;
    j->wsum = i->wsum + wsum;
    j->vect = (i - i1); /* number of choice */
    j++;
  }
  mid->lset = j - 1;
  mid->size = SIZE(mid);
  solution.size = 1;
  solution.vmax = mid->size;
  solution.list[0].size = old->size;
  solution.list[0].set  = old;
}


/* ======================================================================
				partsort
   ====================================================================== */

void partsort(ordstack *stack, ordrec *f, ordrec *l)
{
  register itype mp, mw;
  register ordrec *i, *j, *m;
  int d;

  d = l - f + 1;
  if (d <= 1) return;
  m = f + d / 2;
  if (DET(f->dp, f->dw, m->dp, m->dw) < 0) SWAPO(f, m);
  if (d > 2) {
    if (DET(m->dp, m->dw, l->dp, l->dw) < 0) {
      SWAPO(m, l);
      if (DET(f->dp, f->dw, m->dp, m->dw) < 0) SWAPO(f, m);
    }
  }
  if (d <= 3) return;

  mp = m->dp; mw = m->dw; i = f; j = l;
  for (;;) {
    do i++; while (DET(i->dp, i->dw, mp, mw) > 0);
    do j--; while (DET(j->dp, j->dw, mp, mw) < 0);
    if (i > j) break;
    SWAPO(i, j);
  }

  if (stack->optim == MINIMIZE) {
    pushstack(stack, f, i-1); partsort(stack, i, l);
  } else {
    pushstack(stack, i, l); partsort(stack, f, i-1);
  }
}


/* ======================================================================
			       checkinterval
   ====================================================================== */

void checkinterval(ordstack *s)
{
  int l;
  ordintv *top;

  if (s->level == -1) return; /* nothing to pop */
  top = &(s->intv[s->level]);
  if ((top->f <= s->i) && (s->i <= top->l)) {
    /* current i is in next interval */
    (s->level)--;
    partsort(s, top->f, top->l);
  }
}


/* ======================================================================
				 defineedges
   ====================================================================== */

void defineedges(ordstack *stacka, ordstack *stackb, isetset *head)
{
  register itemrec *i;
  register itype p1, w1;
  itemrec *i1, *im, *m1, *m2, d1, d2;
  itemset *j, *jm;
  ordrec *a, *b;

  jm = head->lset; a = stacka->first; b = stackb->last;
  for (j = head->fset+1; j <= jm; j++, a++, b--) {
    i1 = j->fset;   im = j->lset;
    p1 = j->f.psum; w1 = j->f.wsum;
    d1.psum = p1; d1.wsum = w1+1; m1 = &d1;
    d2.psum = p1-1; d2.wsum = w1; m2 = &d2;
    for (i = i1+1; i <= im; i++) {
      if (i->wsum > w1) {
	if (DET(i->psum-p1,i->wsum-w1,m1->psum-p1,m1->wsum-w1) > 0) m1 = i;
      }
      if (i->wsum < w1) {
	if (DET(p1-i->psum,w1-i->wsum,p1-m2->psum,w1-m2->wsum) < 0) m2 = i;
      }
    }
    a->dp = m1->psum - p1; a->dw = m1->wsum - w1; a->ref = j;
    b->dp = p1 - m2->psum; b->dw = w1 - m2->wsum; b->ref = j;
  }
  a->dp = 0; a->dw = 1; a->ref= NULL;
  b->dp = 1; b->dw = 0; b->ref= NULL;

  partsort(stacka, stacka->first, stacka->last-1);
  partsort(stackb, stackb->first+1, stackb->last);
}


/* ======================================================================
				 makestacks
   ====================================================================== */

void makestacks(ordstack *stacka, ordstack *stackb, isetset *head)
{
  stacka->first = palloc(head->size * (long) sizeof(ordrec));
  stackb->first = palloc(head->size * (long) sizeof(ordrec));
  stacka->last  = stacka->first + head->size - 1;
  stackb->last  = stackb->first + head->size - 1;
  stacka->level = -1;
  stackb->level = -1;
  stacka->optim = MAXIMIZE;
  stackb->optim = MINIMIZE;
  stacka->i     = stacka->first;
  stackb->i     = stackb->last;
}


/* ======================================================================
				 freestacks
   ====================================================================== */

void freestacks(ordstack *stacka, ordstack *stackb)
{
  pfree(stacka->first);
  pfree(stackb->first);
}


/* ======================================================================
				 minmcknap
   ====================================================================== */

void minmcknap(ntype k, ntype n, itype r, int type)
{
  allinfo a;
  ordstack stacka, stackb;
  stype cstar, psum, wsum, z, c, ub;
  partset mid;
  isetset head;
  itemset *s;
  itype pb, wb;
  boolean optimal;
  int i;

  a.k        = k;
  a.n        = n;
  a.range    = r;
  a.type     = type;

  /* make test example */
  cstar = inititems(&a, &head, k, n, r, type);

  a.summul   = 0;
  a.maxmul   = 0;
  a.antmul   = 0;
  a.redusets = 0;
  a.reduitems= 0;
  a.redukill = 0;
  a.domikill = 0;

  starttime();

  c = cstar;
  makestacks(&stacka, &stackb, &head);
  partition(&a, &head, c, &psum, &wsum, &pb, &wb);
  restore(&head, psum, wsum);
  endtime(&a.timepar);
  if (psum == a.dantzig) {
    stacka.i->dp = pb; stacka.i->dw = wb; stacka.i->ref = head.fset;
    stackb.i->dp = pb; stackb.i->dw = wb; stackb.i->ref = head.fset;
  } else {
    defineedges(&stacka, &stackb, &head);
  }
  endtime(&a.timesort);

  z = psum-1; ub = a.dantzig;
  for (i = 1; ; i++) {
    initfirst(&mid, head.fset, psum, wsum);
    for (;;) {
      reduceset(&mid, stacka.i, stackb.i, &z, c);
      if ((mid.size == 0) || (z == ub)) break;

      s = stacka.i->ref; if (s == NULL) break;
      (stacka.i)++; checkinterval(&stacka);
      if (!s->used) {
	reduceitem(&a, s, psum, wsum, pb, wb, z, c);
	domiitem(s);
	if (s->size > 1) {
	  multiply(&a, &mid, s);
	  a.antmul++;
	  a.summul += mid.size;
	  if (mid.size > a.maxmul) a.maxmul = mid.size;
	}
	s->used = TRUE;
      }

      reduceset(&mid, stacka.i, stackb.i, &z, c);
      if ((mid.size == 0) || (z == ub)) break;

      s = stackb.i->ref; if (s == NULL) break;
      (stackb.i)--; checkinterval(&stackb);
      if (!s->used) {
	reduceitem(&a, s, psum, wsum, pb, wb, z, c);
	domiitem(s);
	if (s->size > 1) {
	  multiply(&a, &mid, s);
	  a.antmul++;
	  a.summul += mid.size;
	  if (mid.size > a.maxmul) a.maxmul = mid.size;
	}
	s->used = TRUE;
      }
    }
    if (i == 1) a.zstar = z;
    optimal = definesol(&a, psum, wsum, &stacka, &stackb, &c, &z, &ub);
    pfree(mid.fset);
    if (optimal) { a.iterates += i; break; }
    vis(1,"TUR TIL MED c %ld, z %ld, ub %ld\n", c, z, ub);
    if (i > 10) error("for mange runder");
  }

  a.gap = a.dantzig - a.zstar;
  freestacks(&stacka, &stackb);

  endtime(&a.time);
  checksolution(&a, &head, a.zstar, cstar);
  sumdata(&a);
  freeitems(&head);
}


/* ======================================================================
				    main
   ====================================================================== */

void main(int argc, char *argv[])
{
  int n, r, k, type, v;
  stype c;
  isetset head;

  if (argc == 5) {
    k = atoi(argv[1]);
    n = atoi(argv[2]);
    r = atoi(argv[3]);
    type = atoi(argv[4]);
    printf("Mcknap %d %d %d %d\n", k, n, r, type);
  } else {
    printf("Mcknap\n");
    printf("k = "); scanf("%d", &k);
    printf("n = "); scanf("%d", &n);
    printf("r = "); scanf("%d", &r);
    printf("t = "); scanf("%d", &type);
  }

  inittrace("mc");
  fprintf(trace,"\nMCKNAP: k: %d, n: %d, r: %d, type: %d\n", k, n, r, type);

  for (v = START; v <= TESTS; v++) {
    srand(v);

    vis(2,"\nPROBLEM: k=%d, n=%d, r=%d, type=%d, v=%d\n", k, n, r, type, v);
    vis(2,"==========================================\n");

    checktime();
    minmcknap(k, n, r, type);
  }

  sumdata(NULL);
  closetrace();
}


