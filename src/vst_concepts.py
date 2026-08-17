"""
Visualize all Variance Stabilizing Transformation (VST) concepts on real data.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings("ignore")

df = pd.read_pickle('data_part2.pkl')
price = df['price'].values

# ── The transforms ──
def signed_log(p):   return np.sign(p)*np.log1p(np.abs(p))     # yours
def asinh(p):        return np.arcsinh(p)                       # field standard = log(p+sqrt(p^2+1))
def plain_log(p):    return np.log(p)                           # fails on negatives

BG="#f5f4f0"; ABG="#eeede9"
fig = plt.figure(figsize=(20,22)); fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(4,2,figure=fig,hspace=0.38,wspace=0.24,top=0.95,bottom=0.04,left=0.07,right=0.96)

# ═══ PANEL 1: Why plain log FAILS on negative prices ═══
ax = fig.add_subplot(gs[0,0]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
x = np.linspace(-100, 300, 500)
ax.plot(x, np.where(x>0, np.log(np.maximum(x,1e-9)), np.nan), color='#D85A30', lw=2.5, label='plain log(p)')
ax.axvspan(-100, 0, color='#FCEBEB', alpha=0.6)
ax.text(-50, 3, 'UNDEFINED\n(prices go\nnegative here)', ha='center', fontsize=11, color='#791F1F', fontweight='bold')
ax.axvline(0, color='#333', lw=0.8, ls=':')
ax.set_title('1. Why plain log FAILS\nlog of a negative number does not exist', fontsize=12, fontweight='bold')
ax.set_xlabel('price (€/MWh)'); ax.set_ylabel('log(price)')
ax.legend(fontsize=10); ax.grid(alpha=0.2)

# ═══ PANEL 2: The transforms side by side ═══
ax = fig.add_subplot(gs[0,1]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
xr = np.linspace(-500, 936, 600)
ax.plot(xr, signed_log(xr), color='#1D9E75', lw=2.5, label='signed-log (yours)')
ax.plot(xr, asinh(xr), color='#0C447C', lw=2.0, ls='--', label='asinh (field standard)')
ax.axhline(0, color='#999', lw=0.5); ax.axvline(0, color='#999', lw=0.5)
ax.set_title('2. Both transforms handle negatives & spikes\nsigned-log and asinh are almost identical', fontsize=12, fontweight='bold')
ax.set_xlabel('price (€/MWh)'); ax.set_ylabel('transformed value')
ax.legend(fontsize=10); ax.grid(alpha=0.2)

# ═══ PANEL 3: Raw price distribution (wild) ═══
ax = fig.add_subplot(gs[1,0]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
ax.hist(price, bins=100, color='#D85A30', alpha=0.85, edgecolor='white', lw=0.2)
ax.set_yscale('log')
ax.set_title(f'3. RAW price — wild range\nstd = {price.std():.0f}, spikes to €{price.max():.0f}', fontsize=12, fontweight='bold')
ax.set_xlabel('price (€/MWh)'); ax.set_ylabel('count (log scale)')
ax.axvline(0, color='#333', lw=0.8, ls=':')

# ═══ PANEL 4: After asinh (tamed) ═══
ax = fig.add_subplot(gs[1,1]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
ax.hist(asinh(price), bins=100, color='#1D9E75', alpha=0.85, edgecolor='white', lw=0.2)
ax.set_title(f'4. After asinh — tamed & symmetric\nstd = {asinh(price).std():.2f}, easy to model', fontsize=12, fontweight='bold')
ax.set_xlabel('asinh(price)'); ax.set_ylabel('count')
ax.axvline(0, color='#333', lw=0.8, ls=':')

# ═══ PANEL 5: signed-log vs asinh difference (tiny) ═══
ax = fig.add_subplot(gs[2,0]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
diff = signed_log(xr) - asinh(xr)
ax.plot(xr, diff, color='#7F77DD', lw=2)
ax.axhline(0, color='#999', lw=0.5)
ax.fill_between(xr, diff, alpha=0.2, color='#7F77DD')
ax.set_title('5. Difference: signed-log minus asinh\nnearly zero everywhere — they agree', fontsize=12, fontweight='bold')
ax.set_xlabel('price (€/MWh)'); ax.set_ylabel('difference')
ax.grid(alpha=0.2)

# ═══ PANEL 6: The two ORDERS (yours vs field) ═══
ax = fig.add_subplot(gs[2,1]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
# Your order: signed-log THEN standardize
a = signed_log(price); a_std = (a - a.mean())/a.std()
# Field order: standardize THEN asinh
s = (price - price.mean())/price.std(); s_asinh = asinh(s)
ax.hist(a_std, bins=80, color='#1D9E75', alpha=0.5, label='yours: log then standardize', density=True)
ax.hist(s_asinh, bins=80, color='#0C447C', alpha=0.5, label='field: standardize then asinh', density=True)
ax.set_title('6. Two recipe ORDERS compared\nboth give a tidy, model-friendly shape', fontsize=12, fontweight='bold')
ax.set_xlabel('transformed value'); ax.set_ylabel('density')
ax.legend(fontsize=9); ax.grid(alpha=0.2)

# ═══ PANEL 7: Transform helps MORE during volatile periods ═══
ax = fig.add_subplot(gs[3,0]); ax.set_facecolor(ABG); ax.spines[['top','right']].set_visible(False)
# Split into calm vs volatile by rolling std, show raw range vs transformed range
roll_std = pd.Series(price).rolling(168,min_periods=24).std().bfill().values
calm = roll_std < np.quantile(roll_std, 0.5)
vol  = roll_std >= np.quantile(roll_std, 0.85)
cats = ['Calm\nperiods','Volatile\nperiods']
raw_spread = [price[calm].std(), price[vol].std()]
tr_spread  = [asinh(price[calm]).std(), asinh(price[vol]).std()]
x = np.arange(2); w=0.35
ax.bar(x-w/2, raw_spread, w, color='#D85A30', alpha=0.85, label='raw price spread')
ax2 = ax.twinx()
ax2.bar(x+w/2, tr_spread, w, color='#1D9E75', alpha=0.85, label='asinh spread')
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel('raw price std (€)', color='#D85A30'); ax2.set_ylabel('asinh std', color='#1D9E75')
ax.set_title('7. Why transforms help MOST in volatile times\nraw spread explodes; asinh spread stays controlled', fontsize=12, fontweight='bold')

# ═══ PANEL 8: The pipeline diagram ═══
ax = fig.add_subplot(gs[3,1]); ax.axis('off'); ax.set_facecolor(BG)
ax.text(0.5, 0.95, 'The recommended recipe', ha='center', fontsize=13, fontweight='bold', transform=ax.transAxes)
steps = [
    ('Raw price', '€-500 to €936', '#D85A30'),
    ('Standardize', '(p - mean)/std', '#BA7517'),
    ('asinh transform', 'log(y+sqrt(y²+1))', '#0C447C'),
    ('Model-ready', 'tidy ~[-4,+4]', '#1D9E75'),
]
y = 0.78
for i,(t,d,c) in enumerate(steps):
    ax.add_patch(plt.Rectangle((0.15,y-0.06),0.7,0.11,transform=ax.transAxes,color=c,alpha=0.85))
    ax.text(0.5,y,f'{t}',ha='center',va='center',transform=ax.transAxes,color='white',fontsize=11,fontweight='bold')
    ax.text(0.5,y-0.035,d,ha='center',va='center',transform=ax.transAxes,color='white',fontsize=8)
    if i<len(steps)-1:
        ax.annotate('',xy=(0.5,y-0.10),xytext=(0.5,y-0.06),transform=ax.transAxes,
                    arrowprops=dict(arrowstyle='->',color='#555',lw=1.5))
    y -= 0.19
ax.text(0.5, 0.02, 'Uniejewski, Weron & Ziel (2018) — the foundational VST paper',
        ha='center', fontsize=8.5, style='italic', color='#666', transform=ax.transAxes)

fig.suptitle('Variance Stabilizing Transformations (VSTs) for Electricity Prices — All Concepts on Real SMARD Data',
             fontsize=16, fontweight='bold', y=0.975)
plt.savefig('vst_concepts.png', dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print("saved vst_concepts.png")

# Print a quick numeric summary
print(f"\nRaw price:      std={price.std():.1f}, range={price.max()-price.min():.0f}")
print(f"signed-log:     std={signed_log(price).std():.3f}")
print(f"asinh:          std={asinh(price).std():.3f}")
print(f"max |signed-log - asinh| = {np.abs(signed_log(price)-asinh(price)).max():.4f}  (nearly identical)")
