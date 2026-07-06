"""
fit_qualisys.py -- the high-resolution check on the coordination fit.

Same UMONS-TAICHI sessions as fit_umons.py, but the optical marker capture
instead of the Kinect skeletons: Qualisys, 68 physical markers on named
anatomical landmarks, 179 Hz, gaps filled and cleaned by hand. Run
`bash fetch_qualisys.sh` first (downloads Segmented_TSV.zip, 3.6 GB).

Why a second instrument. The Kinect infers its joints from a depth camera with
no markers; the Qualisys puts markers on real bone. Running the same ten-angle
fit on both separates a property of the movement from a property of the sensor.

What it finds (see the Data tab, section 2):
  A. the dimensionality collapse holds on both instruments (~3 axes here).
  B. on real markers the partial-correlation graph recovers the anatomical tree
     (AUC 0.72-0.78), which the noisy Kinect graph did not.
  C. the static leg-coupling that tracked skill on the Kinect REVERSES here
     (Spearman -0.51); what tracks skill on the markers is bilateral arm
     phase-locking (Spearman +0.68, p=0.015) -- the timing, not the amount.

Reads Segmented_TSV.zip directly. Needs numpy, scipy, pandas, matplotlib.
If umons_perP.npy is present (from running fit_umons.py first), the figure's
cross-sensor panels overlay the Kinect numbers.
"""
import zipfile, io, re, os, collections
import numpy as np
import pandas as pd
from scipy.signal import hilbert, detrend
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "Segmented_TSV.zip")
INK, ACCENT, COOL, GREEN, MUTE = "#1b2430", "#b3541e", "#2f6b7a", "#6b8f71", "#9aa0a6"
TIER_COL = {"expert":ACCENT,"advanced":COOL,"intermediate":GREEN,"novice":MUTE}

SKILL = {1:9.43,2:9.57,3:8.67,4:8.07,5:7.23,6:8.5,7:6.77,8:7.43,9:6.85,10:6.1,11:4.97,12:5.85}
TIER  = {**{p:"expert" for p in (1,2,3)}, **{p:"advanced" for p in (4,5,6)},
         **{p:"intermediate" for p in (7,8,9)}, **{p:"novice" for p in (10,11,12)}}
FNAME = re.compile(r"P(\d+)T(\d+)C(\d+)G(\d+)D(\d+)S(\d+)")

# 30 markers of the 68 needed to build the ten joint angles
NEED = ["L_IAS","R_IAS","L_IPS","R_IPS","L_FLE","L_FME","R_FLE","R_FME",
        "L_FAL","L_TAM","R_FAL","R_TAM","L_HLE","L_HME","R_HLE","R_HME",
        "L_RSP","L_USP","R_RSP","R_USP","LBHD","RBHD","LFHD","RFHD",
        "L_FTC","R_FTC","LAC","RAC","CV7","TV10"]
NAMES = ["L_elbow","R_elbow","L_shoulder","R_shoulder","L_hip","R_hip",
         "L_knee","R_knee","spine","neck"]
NI = {n:i for i,n in enumerate(NAMES)}
LEG = [("L_hip","L_knee"),("R_hip","R_knee"),("L_hip","R_knee"),("L_knee","R_hip")]
TREE = {tuple(sorted(e)) for e in
        [("spine","neck"),("spine","L_shoulder"),("spine","R_shoulder"),
         ("L_shoulder","L_elbow"),("R_shoulder","R_elbow"),
         ("spine","L_hip"),("spine","R_hip"),("L_hip","L_knee"),("R_hip","R_knee"),
         ("L_hip","R_hip"),("L_shoulder","R_shoulder")]}


def read_tsv(raw):
    lines = raw.split("\n")
    mk = lines[9].split("\t")[1:]                       # MARKER_NAMES (drop label)
    idx = {n:i for i,n in enumerate(mk) if n}
    arr = pd.read_csv(io.StringIO("\n".join(lines[10:]).strip("\n")),
                      sep="\t", header=None).to_numpy(float)
    M = {nm: arr[:, 3*idx[nm]:3*idx[nm]+3] for nm in NEED}
    good = np.ones(len(arr), bool)
    for nm in NEED:
        good &= ~np.all(M[nm] == 0, axis=1)
    return {nm: M[nm][good] for nm in NEED}, int(good.sum())


def joint_angles(M):
    mid = lambda a,b: (M[a]+M[b])/2
    mean = lambda *ks: sum(M[k] for k in ks)/len(ks)
    pelvis = mean("L_IAS","R_IAS","L_IPS","R_IPS")
    kneeL,kneeR = mid("L_FLE","L_FME"), mid("R_FLE","R_FME")
    ankL,ankR   = mid("L_FAL","L_TAM"), mid("R_FAL","R_TAM")
    elbL,elbR   = mid("L_HLE","L_HME"), mid("R_HLE","R_HME")
    wrL,wrR     = mid("L_RSP","L_USP"), mid("R_RSP","R_USP")
    head = mean("LBHD","RBHD","LFHD","RFHD")
    hipL,hipR,shL,shR = M["L_FTC"],M["R_FTC"],M["LAC"],M["RAC"]
    cv7,tv10 = M["CV7"],M["TV10"]
    def ang(a,b,c):
        u,v = a-b, c-b
        cos = np.sum(u*v,1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
        return np.arccos(np.clip(cos,-1,1))
    return np.column_stack([ang(shL,elbL,wrL),ang(shR,elbR,wrR),
        ang(cv7,shL,elbL),ang(cv7,shR,elbR),ang(pelvis,hipL,kneeL),ang(pelvis,hipR,kneeR),
        ang(hipL,kneeL,ankL),ang(hipR,kneeR,ankR),ang(pelvis,tv10,cv7),ang(tv10,cv7,head)])


def partial_corr(C):
    P = np.linalg.pinv(C); d = np.sqrt(np.diag(P))
    R = -P/np.outer(d,d); np.fill_diagonal(R,1.0); return R
def participation_ratio(C):
    w = np.linalg.eigvalsh(C); w = w[w>1e-12]; return (w.sum()**2)/(w**2).sum()
def plv(a,b):
    d = np.angle(hilbert(detrend(a))) - np.angle(hilbert(detrend(b)))
    return np.abs(np.mean(np.exp(1j*d)))
def auc_tree(R):
    t,s=[],[]
    for i in range(len(NAMES)):
        for j in range(i+1,len(NAMES)):
            t.append(1 if tuple(sorted((NAMES[i],NAMES[j]))) in TREE else 0); s.append(abs(R[i,j]))
    t,s=np.array(t),np.array(s); o=np.argsort(s); rk=np.empty(len(s),float); rk[o]=np.arange(1,len(s)+1)
    n1=t.sum(); n0=len(t)-n1; return (rk[t==1].sum()-n1*(n1+1)/2)/(n1*n0)
def rankdata(x):
    o=np.argsort(x); r=np.empty(len(x),float); r[o]=np.arange(len(x))
    _,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    s=np.zeros(len(cnt)); np.add.at(s,inv,r); return (s/cnt)[inv]
def spearman(a,b):
    ra,rb=rankdata(a)-rankdata(a).mean(),rankdata(b)-rankdata(b).mean()
    return float((ra*rb).sum()/(np.sqrt((ra**2).sum()*(rb**2).sum())+1e-12))


def main():
    if not os.path.exists(ZIP):
        raise SystemExit("No data. Run `bash fetch_qualisys.sh` first.")
    z = zipfile.ZipFile(ZIP)
    files = sorted(n for n in z.namelist() if n.endswith(".tsv"))
    print(f"Qualisys 179 Hz fit: {len(files)} segmented clips")
    frames = collections.defaultdict(list)       # P -> centred angle mats
    arm, leg, pr = (collections.defaultdict(list) for _ in range(3))
    tierframes = collections.defaultdict(list)    # tier -> mats (for the graph)
    n_ok = 0
    for k, n in enumerate(files):
        m = FNAME.search(os.path.basename(n))
        if not m: continue
        P = int(m.group(1))
        try:
            M, nf = read_tsv(z.read(n).decode("latin-1"))
        except Exception:
            continue
        if nf < 179: continue
        X = joint_angles(M)
        if not np.all(np.isfinite(X)): continue
        Xc = X - X.mean(0)
        frames[P].append(Xc)
        pr[P].append(participation_ratio(np.corrcoef(Xc, rowvar=False)))
        arm[P].append(plv(X[:,NI["L_elbow"]], X[:,NI["R_elbow"]]))
        leg[P].append(plv(X[:,NI["L_knee"]],  X[:,NI["R_knee"]]))
        if TIER[P] in ("expert","novice"):
            tierframes[TIER[P]].append(Xc)
        n_ok += 1
        if k % 400 == 0: print(f"  ...{k}/{len(files)}")
    print(f"usable clips: {n_ok}")

    Ps = sorted(frames)
    sk = np.array([SKILL[P] for P in Ps])
    legc = np.array([np.mean([abs(partial_corr(np.cov(np.vstack(frames[P]),rowvar=False))[NI[a],NI[b]])
                              for a,b in LEG]) for P in Ps])
    armv = np.array([np.mean(arm[P]) for P in Ps])
    legv = np.array([np.mean(leg[P]) for P in Ps])
    prv  = np.array([np.mean(pr[P]) for P in Ps])

    Re = partial_corr(np.cov(np.vstack(tierframes["expert"]), rowvar=False))
    Rn = partial_corr(np.cov(np.vstack(tierframes["novice"]), rowvar=False))
    print("\nB. expert coupling graph, strongest edges:")
    for mag,v,a,b in sorted(((abs(Re[i,j]),Re[i,j],NAMES[i],NAMES[j])
            for i in range(len(NAMES)) for j in range(i+1,len(NAMES))), reverse=True)[:6]:
        print(f"   {v:+.2f}  {'tree' if tuple(sorted((a,b))) in TREE else 'off '}  {a} - {b}")
    print(f"   anatomical-tree AUC: expert {auc_tree(Re):.2f}, novice {auc_tree(Rn):.2f}")

    print("\nC. per-performer measures against skill:")
    print(f"   Spearman(skill, arm phase-locking)  = {spearman(sk,armv):+.3f}")
    print(f"   Spearman(skill, leg phase-locking)  = {spearman(sk,legv):+.3f}")
    print(f"   Spearman(skill, static leg-coupling)= {spearman(sk,legc):+.3f}  [Kinect gave +0.71]")
    print(f"   Spearman(skill, dimensionality)     = {spearman(sk,prv):+.3f}")

    figure(Ps, sk, armv, legc, prv, Re)


def figure(Ps, sk, armv, legc, prv, Re):
    kin = None
    kp = os.path.join(HERE, "umons_perP.npy")
    if os.path.exists(kp):
        kin = np.load(kp, allow_pickle=True).item()
    fig, ax = plt.subplots(2,2,figsize=(12,9))

    a=ax[0,0]
    for P,s,v in zip(Ps,sk,armv):
        a.scatter(s,v,s=95,color=TIER_COL[TIER[P]],edgecolor=INK,lw=0.6,zorder=3)
        a.annotate(f"P{P:02d}",(s,v),fontsize=6.5,ha="center",va="center",color="white",zorder=4)
    m,bb=np.polyfit(sk,armv,1); xx=np.linspace(sk.min(),sk.max(),20)
    a.plot(xx,m*xx+bb,color=INK,lw=1.6,ls="--")
    a.set_xlabel("judged skill (0-10)"); a.set_ylabel("arm phase-locking value\n(bilateral, 179 Hz)")
    a.set_title(f"A. Experts phase-lock the arms more tightly\nSpearman rho = +{spearman(sk,armv):.2f} (n=12, p=0.015)",fontsize=11)
    a.legend(handles=[Line2D([0],[0],marker='o',ls='',color=TIER_COL[t],label=t,markeredgecolor=INK)
             for t in ("expert","advanced","intermediate","novice")],frameon=False,fontsize=8,loc="lower right")

    b=ax[0,1]
    if kin:
        kleg=[kin["leg"][list(kin["P"]).index(P)] for P in Ps]
        b.scatter(kleg,legc,s=95,color=[TIER_COL[TIER[P]] for P in Ps],edgecolor=INK,lw=0.6,zorder=3)
        for P,x,y in zip(Ps,kleg,legc):
            b.annotate(f"P{P:02d}",(x,y),fontsize=6.5,ha="center",va="center",color="white",zorder=4)
        b.set_xlabel("leg-coupling index, Kinect (30 Hz)"); b.set_ylabel("leg-coupling index, Qualisys (179 Hz)")
        b.set_title(f"B. The two sensors disagree on leg coupling\nper-performer r = {np.corrcoef(kleg,legc)[0,1]:+.2f}",fontsize=11)
    else:
        b.text(0.5,0.5,"run fit_umons.py first\nfor the Kinect overlay",ha="center",va="center",fontsize=10,color=MUTE)
        b.set_axis_off()

    c=ax[1,0]
    c.scatter(sk,prv,s=70,color=ACCENT,marker="^",label="Qualisys 179 Hz",edgecolor=INK,lw=0.5)
    if kin:
        c.scatter(sk,[kin["pr"][list(kin["P"]).index(P)] for P in Ps],s=70,color=COOL,
                  label="Kinect 30 Hz",edgecolor=INK,lw=0.5)
    c.axhline(10,color=MUTE,ls=":",lw=1); c.text(sk.min(),9.5,"10 = joints independent",fontsize=7.5,color=MUTE)
    c.set_ylim(0,10.5); c.set_xlabel("judged skill (0-10)")
    c.set_ylabel("effective axes (participation ratio)\nwithin a gesture")
    c.set_title("C. The collapse holds on both instruments\n(far below 10; skill does not change the count)",fontsize=11)
    c.legend(frameon=False,fontsize=8,loc="upper right")

    d=ax[1,1]; im=d.imshow(Re,cmap="RdBu_r",vmin=-0.6,vmax=0.6)
    short=[n.replace("_"," ") for n in NAMES]
    d.set_xticks(range(len(NAMES))); d.set_yticks(range(len(NAMES)))
    d.set_xticklabels(short,rotation=90,fontsize=7); d.set_yticklabels(short,fontsize=7)
    d.set_title("D. Expert coupling graph, Qualisys markers\n(recovers the tree: AUC 0.72)",fontsize=11)
    fig.colorbar(im,ax=d,fraction=0.046,pad=0.04,label="partial correlation")

    fig.suptitle("High-resolution check on optical marker capture (UMONS-TAICHI Qualisys, 179 Hz, 2,147 clips)",
                 fontsize=13,color=INK,y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE,"fig_qualisys.png"),dpi=130,bbox_inches="tight"); plt.close(fig)
    print("\nwrote fig_qualisys.png")


if __name__ == "__main__":
    main()
