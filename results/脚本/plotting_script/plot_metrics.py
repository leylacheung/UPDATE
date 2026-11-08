from pathlib import Path
import json, zipfile, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator, NullLocator

import argparse
parser = argparse.ArgumentParser(description='Plot PolymerBenchmark2026 test RMSE and R2 comparisons.')
parser.add_argument('--input', type=Path, required=True, help='CSV or tab-separated results table')
parser.add_argument('--output', type=Path, default=Path('figures'), help='Output directory')
args = parser.parse_args()
SOURCE = args.input
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SOURCE, sep=None, engine='python')
test=df[df.split=='test'].copy()
models=['RDKit_RF','polyBERT','polymer_chemprop','polymer_periodic_graph']
labels=['RDKit RF','polyBERT','Polymer Chemprop','Periodic graph']
colors=['#0072B2','#E69F00','#009E73','#CC79A7']
markers=['o','s','^','D']
experiments=[
 'MD_5000/Cp/alternating_Cp/seed_42',
 'MD_5000/density/alternating_density/seed_42',
 'MD_5000/refractive_index/alternating_refractive_index/seed_42',
 'MD_5000/Rg/alternating_Rg/seed_42',
 'PolyMetriX/Tg/Tg/seed_42',
 'MD_5000/Cp/homopolymer_Cp/seed_42',
 'MD_5000/Cv/homopolymer_Cv/seed_42',
 'MD_5000/density/homopolymer_density/seed_42',
 'MD_5000/refractive_index/homopolymer_refractive_index/seed_42',
 'MD_5000/Rg/homopolymer_Rg/seed_42']
titles=['Cp · alternating','Density · alternating','Refractive index · alternating','Rg · alternating','Tg · homopolymer (PolyMetriX)',
        'Cp · homopolymer','Cv · homopolymer','Density · homopolymer','Refractive index · homopolymer','Rg · homopolymer']
assert len(test)==116 and set(test.experiment)==set(experiments)
assert (test.n_folds==5).all() and (test.r2_n_folds==5).all()
assert not test.duplicated(['experiment','model','train_size']).any()
assert np.isfinite(test[['rmse_mean','rmse_std','r2_mean','r2_std']]).all().all()
assert (test[['rmse_std','r2_std']]>=0).all().all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#8A929C','axes.labelcolor':'#263445','text.color':'#263445',
 'xtick.color':'#526071','ytick.color':'#526071','pdf.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white'})
handles=[Line2D([0],[0],color=c,marker=m,lw=1.8,label=l) for c,m,l in zip(colors,markers,labels)]
global_r2_low=min(-0.05,float((test.r2_mean-test.r2_std).min())-.1)

def panel(ax,sub,metric,common_r2=False):
    for model,color,marker in zip(models,colors,markers):
        s=sub[sub.model==model].sort_values('train_size')
        assert len(s)==sub.train_size.nunique()
        ax.errorbar(s.train_size,s[metric+'_mean'],yerr=s[metric+'_std'],
                    color=color,marker=marker,markersize=4.5,lw=1.8,
                    elinewidth=.8,capsize=2.5,alpha=.95)
    sizes=sorted(sub.train_size.unique())
    ax.set_xscale('log');ax.set_xticks(sizes);ax.set_xticklabels([f'{n:,}' for n in sizes])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(sizes)*.8,max(sizes)*1.23)
    ax.grid(axis='y',color='#E3E8EF',lw=.7);ax.set_axisbelow(True)
    ax.set_xlabel('Training samples (log scale)')
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if metric=='rmse':
        ax.set_ylim(0,float((sub.rmse_mean+sub.rmse_std).max())*1.12)
        ax.set_ylabel('RMSE (original target units) ↓')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
    else:
        low=global_r2_low if common_r2 else min(0,float((sub.r2_mean-sub.r2_std).min()))-.08
        ax.set_ylim(low,1.06);ax.axhline(0,color='#8491A1',lw=.8,ls='--')
        if common_r2:
            ax.set_yticks([-1.5,-1.0,-0.5,0,0.5,1.0])
        elif low > -0.1:
            ax.set_yticks([0,0.25,0.5,0.75,1.0])
        ax.set_ylabel('R² ↑')

def overview(metric):
    fig,axs=plt.subplots(2,5,figsize=(19,8.4))
    for i,(e,title,ax) in enumerate(zip(experiments,titles,axs.flat)):
        panel(ax,test[test.experiment==e],metric,common_r2=True)
        ax.set_title(f'{chr(97+i)}) {title}',loc='left',pad=12)
    title='Test RMSE — lower is better' if metric=='rmse' else 'Test R² — higher is better'
    fig.suptitle(title,x=.04,y=.98,ha='left',fontsize=21,fontweight='bold')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.94),ncol=4,frameon=False)
    scale='RMSE axes use condition-specific scales; do not compare magnitudes across properties.' if metric=='rmse' else 'All overview panels share the same R² scale; negative values are retained.'
    fig.text(.04,.035,'Test only · 5 repeated splits · mean ± population SD (not confidence intervals). '+scale,fontsize=10)
    fig.text(.04,.012,'R² is recovered from published fold errors and verified labels; these are not new training runs. All panels: MD_5000 unless labeled PolyMetriX.',fontsize=9,color='#526071')
    fig.subplots_adjust(left=.055,right=.985,bottom=.14,top=.80,wspace=.38,hspace=.52)
    fig.savefig(OUT/f'{metric}_overview.png',dpi=200)
    fig.savefig(OUT/f'{metric}_overview.svg')
    return fig

with PdfPages(OUT/'RMSE_R2_comparison.pdf') as pdf:
    for metric in ['rmse','r2']:
        fig=overview(metric);pdf.savefig(fig);plt.close(fig)
    for i,(e,title) in enumerate(zip(experiments,titles)):
        sub=test[test.experiment==e]
        fig,axs=plt.subplots(1,2,figsize=(12,5.4))
        panel(axs[0],sub,'rmse');panel(axs[1],sub,'r2')
        axs[0].set_title('RMSE · lower is better',loc='left')
        axs[1].set_title('R² · higher is better',loc='left')
        fig.suptitle(title+' | '+e.split('/')[0],x=.07,y=.98,ha='left',fontsize=17,fontweight='bold')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.91),ncol=4,frameon=False)
        fig.text(.07,.045,'Test only · 5 folds · mean ± population SD · R² recovered from published errors; no retraining.',fontsize=9)
        fig.text(.07,.015,'R² detail panels use local scales and retain zero. RMSE is in the source target units.',fontsize=9,color='#526071')
        fig.subplots_adjust(left=.09,right=.97,bottom=.19,top=.75,wspace=.28)
        pdf.savefig(fig);fig.savefig(OUT/f'condition_{i+1:02d}.png',dpi=180);plt.close(fig)

test.to_csv(OUT/'plotted_test_results.csv',index=False)
summary={'input_rows':len(df),'plotted_test_rows':len(test),'conditions':10,'models':4,'folds':5,
         'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'pdf_pages':12,
         'r2_min_mean':float(test.r2_mean.min()),'error_bars':'population standard deviation, not confidence intervals'}
(OUT/'checks.json').write_text(json.dumps(summary,indent=2))
(OUT/'读图说明.txt').write_text('图中仅使用此次粘贴表格的 test 数据，共116行，10个条件、4个模型。\n横轴为训练样本数（对数刻度）；线和点为5折均值，误差棒为总体标准差，并非置信区间。\nRMSE越低越好；R²越高越好。不同性质的RMSE量纲/尺度不同，不宜跨性质比较大小。原表未标注物理单位，因此标为原始目标单位。\nR²总览使用统一纵轴，保留负值；PDF逐条件详情使用局部纵轴并保留零线。\nR²来自此前对公开逐折误差和原标签的恢复，并非重新训练。\n图中短名称：RDKit RF=RDKit_RF；Polymer Chemprop=polymer_chemprop；Periodic graph=polymer_periodic_graph。\nPDF前两页为总览，后十页为每个条件的RMSE/R²并排对比。\n')
with zipfile.ZipFile(OUT/'全部对比图.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.suffix!='.zip':z.write(p,p.name)
print(json.dumps(summary,indent=2))
