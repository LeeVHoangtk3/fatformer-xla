"""Render the audited paired CIs; run directly beside paired_verification.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render():
    folder=Path(__file__).resolve().parent
    comparison=json.loads((folder/'paired_verification.json').read_text(encoding='utf-8'))['comparison']
    conditions=list(comparison['CFE'])
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,metric,title in zip(axes,['auc','fake_at_real90'],['AUC','Fake recall at Real90']):
        rows=[comparison['CFE'][c] for c in conditions]
        delta=np.array([r['candidate_minus_reference_pp'][metric] for r in rows])
        ci=np.array([r['paired_ci95_pp'][metric] for r in rows])
        ax.errorbar(delta,np.arange(len(rows)),xerr=np.array([delta-ci[:,0],ci[:,1]-delta]),fmt='o',capsize=4,color='#345b82')
        ax.axvline(0,color='#888888',linestyle='--',linewidth=1)
        ax.set_title(title);ax.set_xlabel('P0 minus refitted incumbent (percentage points)')
        ax.grid(axis='x',alpha=.2);ax.set_yticks(np.arange(len(rows)),conditions)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Round 1: matched evaluation on the same 1,600 CFE images',fontsize=13)
    fig.text(.5,.025,'95% paired bootstrap CIs; 1,000 resamples/condition; one training seed; no multiplicity correction.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    fig.savefig(folder/'paired_deltas.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':render()
