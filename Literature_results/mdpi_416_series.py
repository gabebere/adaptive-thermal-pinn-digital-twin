"""Recreate Tables 1-3 of Hsu, Tu & Chang, Axioms 12 (2023), 416.

This single file contains the paper's series formula for its parabolic Example 2
(Equations 108-111), the values printed in the article, and CSV generation.
It is not a finite-difference or Crank-Nicolson solver.

Run:  python mdpi_416_series.py
Output: mdpi_416_tables/*.csv
"""
from pathlib import Path
import csv
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "mdpi_416_tables"
PAPER_TIMES = np.array([0, .1, .2, .4, .6, .8, 1., 1.2])
EXTENDED_TIMES = np.linspace(0, 1.2, 40)  # exactly 40 instances, endpoints included
TERMS = (1, 3, 5, 10, 20)

# Each tuple is (d1,d2,d3,d4), where eta_i(tau)=exp(-d_i*tau).
CASES = {
    "table_1": (1., 1., 1., 1.),
    "table_2": (1., 1., 2., 2.),
    "table_3": (1., 2., 3., 4.),
}

# Values transcribed exactly as printed in the article.
PUBLISHED = {
    "table_1": [[.516,.497,.501,.500,.500],[.229,.246,.243,.243,.243],
                [.174,.189,.187,.187,.187],[.138,.150,.148,.148,.148],
                [.113,.123,.121,.121,.121],[.0921,.100,.0989,.0994,.0994],
                [.0754,.0823,.0810,.0814,.0814],[.0618,.0674,.0663,.0666,.0666]],
    "table_2": [[.516,.497,.501,.500,.500],[.249,.263,.261,.261,.261],
                [.203,.215,.213,.213,.213],[.176,.184,.183,.183,.183],
                [.154,.159,.159,.159,.159],[.133,.136,.136,.136,.136],
                [.113,.115,.115,.115,.115],[.0954,.0969,.0967,.0968,.0968]],
    "table_3": [[.516,.497,.501,.500,.500],[.285,.295,.293,.293,.293],
                [.251,.257,.256,.256,.256],[.229,.230,.230,.230,.230],
                [.202,.201,.202,.202,.202],[.174,.172,.172,.172,.172],
                [.147,.145,.146,.146,.146],[.123,.121,.122,.122,.122]],
}

def boundary_descriptions(decays):
    d1,d2,d3,d4=decays
    return {
        "bc_left": f"theta(0,Y,tau)=(Y-Y^2)*exp(-{d1:g}*tau)",
        "bc_right": f"theta(1,Y,tau)=(Y-Y^2)*exp(-{d2:g}*tau)",
        "bc_bottom": f"theta(X,0,tau)=(X-X^2)*exp(-{d3:g}*tau)",
        "bc_top": f"theta(X,1,tau)=(X-X^2)*exp(-{d4:g}*tau)",
        "initial_condition": "theta(X,Y,0)=(X-X^2)+(Y-Y^2)",
    }

def _convolution_ratio(decay, eigenvalue, tau):
    """Stable (exp(-d*tau)-exp(-lambda*tau))/(lambda-d)."""
    if np.isclose(eigenvalue, decay):
        return tau*np.exp(-eigenvalue*tau)
    return (np.exp(-decay*tau)-np.exp(-eigenvalue*tau))/(eigenvalue-decay)

def center_temperature(tau, terms, decays):
    """Paper Equations (108), (110), (111) at X=Y=0.5, Tr=1."""
    X=Y=.5; total=0.; d1,d2,d3,d4=decays
    for m in range(1,terms+1):
        sm=np.sin(m*np.pi*.5); pm=(-1)**m
        edge_coeff=4*(1-pm)/(m**3*np.pi**3)
        part_a=((1-X)*edge_coeff*np.exp(-d1*tau)+X*edge_coeff*np.exp(-d2*tau))
        part_b=((1-Y)*edge_coeff*np.exp(-d3*tau)+Y*edge_coeff*np.exp(-d4*tau))
        for n in range(1,terms+1):
            sn=np.sin(n*np.pi*.5); pn=(-1)**n
            eigenvalue=(m*m+n*n)*np.pi**2
            prefactor=8*(1-pm)/(m**3*n*np.pi**4)
            def modal(da,db):
                base=(1-pn)*np.exp(-eigenvalue*tau)-np.exp(-da*tau)+pn*np.exp(-db*tau)
                convolution=n*n*np.pi**2*(_convolution_ratio(da,eigenvalue,tau)-pn*_convolution_ratio(db,eigenvalue,tau))
                return prefactor*(base+convolution)
            part_a += sn*modal(d1,d2)
            part_b += sn*modal(d3,d4)
        total += (part_a+part_b)*sm
    return float(total)

def _write(path, times, decays, calculated, published=None):
    metadata=boundary_descriptions(decays)
    fields=["tau",*metadata.keys(),*[f"calculated_terms_{n}" for n in TERMS]]
    if published is not None: fields += [f"published_terms_{n}" for n in TERMS]
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for i,tau in enumerate(times):
            row={"tau":f"{tau:.15g}",**metadata}
            row.update({f"calculated_terms_{n}":f"{calculated[i,j]:.12g}" for j,n in enumerate(TERMS)})
            if published is not None:
                row.update({f"published_terms_{n}":f"{published[i][j]:.12g}" for j,n in enumerate(TERMS)})
            writer.writerow(row)

def generate_tables():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name,decays in CASES.items():
        study=np.array([[center_temperature(t,n,decays) for n in TERMS] for t in PAPER_TIMES])
        dense=np.array([[center_temperature(t,n,decays) for n in TERMS] for t in EXTENDED_TIMES])
        _write(OUTPUT_DIR/f"{name}_study_times.csv",PAPER_TIMES,decays,study,PUBLISHED[name])
        _write(OUTPUT_DIR/f"{name}_40_times.csv",EXTENDED_TIMES,decays,dense)
    readme=OUTPUT_DIR/"README.txt"
    readme.write_text(
        "Each study_times CSV contains both equation-derived values and the exact values printed in the article.\n"
        "Each 40_times CSV evaluates Equations (108)-(111) at 40 equally spaced times from tau=0 to 1.2.\n"
        "All boundary and initial conditions are repeated in every row for self-describing data.\n"
        "Table 1 agrees with the printed values to their reported precision. Tables 2 and 3 do not: the article's\n"
        "printed values are inconsistent with its stated decay constants and Equations (108)-(111).\n",
        encoding="utf-8")
    print(f"Wrote 6 CSV files and README.txt to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    generate_tables()
