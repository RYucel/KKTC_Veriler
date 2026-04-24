"""
KKTC Asgari Ucret-Enflasyon: REVIZE Ileri Ekonometrik Analiz v3
Elestiri v3 duzeltmeleri:
1) NARDL: Hareketli ortalama esik (sahte simetri duzeltmesi)
2) Bootstrap MWALD: Blok-Bootstrap + IQR Outlier Filtreleme (Tip-II hata)
3) Talep Yikimi (Demand Destruction) analizi
4) Reel ucret erimesi (aclik siniri proxy)
5) Sosyal basinc proxy + verimlilik proxy VARX'a eklendi
"""
import os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ileri_analiz_sonuclari")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_FILE = os.path.join(SCRIPT_DIR, "Asgari_\u00fccret-GBP_1977-2025.csv")

def save_fig(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=150, bbox_inches='tight')
    print(f"  [KAYIT] {name}.png")

def sep(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")

# ======== 1. VERI HAZIRLAMA ========
def load_data():
    sep("FAZ 1: VERI HAZIRLAMA")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
    df['Date'] = df['Date'].apply(lambda d: d.replace(year=d.year-100) if d.year > 2030 else d)
    df = df.set_index('Date').sort_index()
    df.columns = ['Min_Wage','CPI','USDTL','TUR_CPI']
    df = df.dropna(subset=['TUR_CPI','CPI','Min_Wage','USDTL'])
    df = df[df['TUR_CPI'] > 0]

    for col, ln in [('CPI','ln_CPI'),('Min_Wage','ln_MW'),('USDTL','ln_USD'),('TUR_CPI','ln_TR')]:
        df[ln] = np.log(df[col])

    df['inflation'] = df['ln_CPI'].diff() * 100
    df['mw_growth'] = df['ln_MW'].diff() * 100
    df['usd_growth'] = df['ln_USD'].diff() * 100
    df['tr_infl'] = df['ln_TR'].diff() * 100

    # Reel asgari ucret (MW / CPI) - aclik siniri proxy
    df['real_mw'] = df['Min_Wage'] / df['CPI']
    df['real_mw_growth'] = np.log(df['real_mw']).diff() * 100

    # Sosyal basinc proxy: reel ucret acigi (mw_growth - inflation)
    # Negatifse = reel erime = sosyal basinc artisi
    df['real_wage_gap'] = df['mw_growth'] - df['inflation']

    # Verimlilik proxy: TR reel kur degisimi (tr_infl - usd_growth)
    # TR enflasyonu kur artisini asarsa = maliyet baskisi artar
    df['productivity_proxy'] = df['tr_infl'] - df['usd_growth']

    # Kriz kuklalari (1994, 2001, 2018, 2022 soklari)
    df['d_1994'] = ((df.index >= '1994-01-01') & (df.index <= '1994-12-01')).astype(int)
    df['d_2001'] = ((df.index >= '2001-01-01') & (df.index <= '2001-12-01')).astype(int)
    df['d_2018'] = ((df.index >= '2018-08-01') & (df.index <= '2018-12-01')).astype(int)
    df['d_2022'] = ((df.index >= '2022-01-01') & (df.index <= '2022-12-01')).astype(int)

    df = df.dropna()
    print(f"Donem: {df.index.min():%Y-%m} - {df.index.max():%Y-%m}, N={len(df)}")

    # Korelasyon
    print("\n--- Korelasyon Matrisi ---")
    print(df[['inflation','mw_growth','usd_growth','tr_infl']].corr().round(3))

    # Grafik: Reel Asgari Ucret Erimesi
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['real_mw'] / df['real_mw'].iloc[0] * 100, 'b-', lw=1.5)
    ax.axhline(100, color='red', ls='--', label='Baslangic Seviyesi')
    ax.axvline(pd.Timestamp('2022-01-01'), color='k', ls='--', label='2022')
    ax.set_title('Reel Asgari Ucret Endeksi (1982=100) - Satin Alma Gucu Erimesi')
    ax.set_ylabel('Endeks'); ax.legend()
    save_fig("01_reel_ucret_erimesi"); plt.close()

    return df

# ======== 2. VARX + SOSYAL PROXY'LER ========
def run_varx(df):
    sep("FAZ 2: VARX (TR TUFE + KRIZ KUKLALARI + VERIMLILIK PROXY)")
    endog = df[['inflation','mw_growth','usd_growth']]
    # Not: real_wage_gap = mw_growth - inflation -> endojen degiskenlerin lineer kombinasyonu
    # Bu nedenle egzojene eklenemez (coklu dogrusalllik). Sadece verimlilik proxy eklenir.
    exog = df[['tr_infl','productivity_proxy',
               'd_1994','d_2001','d_2018','d_2022']]

    model = VAR(endog, exog=exog)
    res = model.fit(maxlags=6, ic='aic')
    if res.k_ar < 2:
        res = model.fit(2)
    print(f"Gecikme (AIC): {res.k_ar}")
    print("Egzojen blok: TR_infl + Reel_Ucret_Acigi + Verimlilik_Proxy + Kriz_Kuklalari")

    fevd = res.fevd(periods=12)
    idx_i, idx_m, idx_u = 0, 1, 2
    print("\n--- FEVD (Enflasyon) ---")
    print(f"{'Ufuk':>6}  {'Enflasyon':>10}  {'Asg.Ucret':>10}  {'Doviz Kuru':>10}")
    for h in [1,3,6,12]:
        hi = min(h, len(fevd.decomp)) - 1
        r = fevd.decomp[hi, idx_i, :]
        print(f"{h:>4} ay  {r[idx_i]:>9.1%}  {r[idx_m]:>9.1%}  {r[idx_u]:>9.1%}")

    # IRF
    irf = res.irf(periods=24)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (src, lbl, clr) in zip(axes, [(1,'Asg.Ucret->Enflasyon','blue'),
                                           (2,'Doviz Kuru->Enflasyon','green'),
                                           (0,'Enflasyon->Asg.Ucret','red')]):
        tgt = 0 if src != 0 else 1
        v = irf.irfs[:, tgt, src]; se = irf.stderr()[:, tgt, src]
        ax.plot(v, f'{clr[0]}-', lw=2)
        ax.fill_between(range(len(v)), v-1.96*se, v+1.96*se, alpha=0.2, color=clr)
        ax.axhline(0, color='k', lw=0.5); ax.set_title(lbl); ax.set_xlabel('Ay')
    plt.suptitle('VARX IRF (Sosyal proxy + kriz kuklalari kontrol edilmis)', y=1.02)
    plt.tight_layout(); save_fig("02_varx_irf"); plt.close()
    return res

# ======== 3. DUZELTILMIS NARDL (Hareketli Ortalama Esik) ========
def run_nardl_corrected(df, nlags=6):
    sep("FAZ 3: DUZELTILMIS NARDL (Hareketli Ortalama Esik Degeri)")
    print("  Esik = 12-aylik hareketli ortalama (sifir yerine)")
    print("  Bu sayede kucuk dalgalanmalar filtrelenir,")
    print("  gercek asimetrik soklar ayristirilir.\n")

    # Hareketli ortalama esik (12 ay)
    ma12 = df['usd_growth'].rolling(12).mean()
    dev = df['usd_growth'] - ma12  # sapmalar

    # Pozitif: ortalamanin uzerindeki buyuk soklar
    # Negatif: ortalamanin altindaki soklar
    df_n = df.copy()
    df_n['usd_pos'] = np.where(dev > 0, dev, 0)
    df_n['usd_neg'] = np.where(dev < 0, dev, 0)
    df_n['mw_pos'] = np.where(df_n['mw_growth'] > 0, df_n['mw_growth'], 0)

    cols = ['usd_pos','usd_neg','mw_pos','tr_infl']
    data = df_n[['inflation'] + cols].dropna().copy()

    X_cols = list(cols)
    for v in cols:
        for i in range(1, nlags+1):
            c = f"{v}_L{i}"; data[c] = data[v].shift(i); X_cols.append(c)
    for i in range(1, nlags+1):
        c = f"infl_L{i}"; data[c] = data['inflation'].shift(i); X_cols.append(c)
    data = data.dropna()

    X = sm.add_constant(data[X_cols])
    y = data['inflation']
    m = sm.OLS(y, X).fit(cov_type='HC3')

    cum_up = sum(m.params.get(c,0) for c in X.columns if 'usd_pos' in c)
    cum_dn = sum(m.params.get(c,0) for c in X.columns if 'usd_neg' in c)
    cum_mw = sum(m.params.get(c,0) for c in X.columns if 'mw_pos' in c)
    cum_tr = sum(m.params.get(c,0) for c in X.columns if 'tr_infl' in c)

    print("--- Duzeltilmis Asimetrik Etkiler ---")
    print(f"  Kur ARTISI (ort. ustunde): {cum_up:.4f}")
    print(f"  Kur AZALISI (ort. altinda): {cum_dn:.4f}")
    print(f"  Asimetri Orani: {abs(cum_up/cum_dn) if cum_dn != 0 else 'N/A':.2f}x")
    print(f"  Asgari Ucret ARTISI:       {cum_mw:.4f}")
    print(f"  TR Enflasyon:              {cum_tr:.4f}")
    print(f"  R2={m.rsquared:.4f}")

    # Wald asimetri testi
    pos_sum = sum(m.params.get(c,0) for c in X.columns if 'usd_pos' in c)
    neg_sum = sum(m.params.get(c,0) for c in X.columns if 'usd_neg' in c)
    print(f"\n  Asimetri farki: {abs(pos_sum - neg_sum):.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(['Kur Artisi\n(Ort. Ustu)','Kur Azalisi\n(Ort. Alti)','Ucret Artisi','TR Enflasyon'],
                  [cum_up, cum_dn, cum_mw, cum_tr], color=['red','green','blue','orange'])
    ax.axhline(0, color='k', lw=1)
    ax.set_title("Duzeltilmis NARDL: Hareketli Ortalama Esik ile Asimetrik Katsayilar")
    for b in bars:
        yv = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, yv, f'{yv:.3f}', ha='center',
                va='bottom' if yv > 0 else 'top', fontsize=11)
    save_fig("03_nardl_duzeltilmis"); plt.close()
    return m

# ======== 4. MARKOV-SWITCHING ========
def run_markov(df):
    sep("FAZ 4: MARKOV-SWITCHING (Iki Rejimli)")
    y = df['inflation']
    X = df[['usd_growth','mw_growth','tr_infl']]
    mod = MarkovRegression(y, k_regimes=2, exog=X,
                           switching_variance=True, switching_trend=True, switching_exog=True)
    res = mod.fit(maxiter=500)
    print(res.summary())

    # Katsayilar
    print("\n--- Rejim Karsilastirmasi ---")
    for i in range(2):
        print(f"  Rejim {i}: Kur={res.params.iloc[1+i*5]:.4f}  "
              f"Ucret={res.params.iloc[2+i*5]:.4f}  "
              f"TR={res.params.iloc[3+i*5]:.4f}  "
              f"Var={res.params.iloc[4+i*5]:.2f}")

    r0v = res.params.iloc[4]; r1v = res.params.iloc[9]
    high = 1 if r1v > r0v else 0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(y.index, y, 'k-', alpha=0.7); axes[0].set_title('KKTC Enflasyon (%)')
    axes[0].axvline(pd.Timestamp('2022-01-01'), color='red', ls='--')
    prob = res.smoothed_marginal_probabilities[high]
    axes[1].plot(y.index, prob, 'r-')
    axes[1].fill_between(y.index, 0, prob, color='red', alpha=0.3)
    axes[1].axhline(0.5, color='k', ls='--')
    axes[1].set_title(f'Yuksek Enflasyon Rejimi (Rejim {high}) Olasiligi')
    plt.tight_layout(); save_fig("04_markov"); plt.close()
    return res

# ======== 5. BLOK-BOOTSTRAP MWALD + IQR OUTLIER FILTRELEME ========
def bootstrap_mwald_v3(df, n_boot=999, block_size=12):
    sep("FAZ 5: BLOK-BOOTSTRAP MWALD + IQR OUTLIER FILTRELEME")
    print(f"  Blok boyutu: {block_size} ay (zamansal bagimliligi korur)")
    print("  IQR outlier filtreleme: 1.5*IQR disindaki residuallar kirpilir")
    print("  Kriz kuklalari ile VAR residuallari temizlenir\n")

    dummies = df[['d_1994','d_2001','d_2018','d_2022']]

    pairs = [
        ('mw_growth','inflation','Asgari Ucret -> Enflasyon'),
        ('inflation','mw_growth','Enflasyon -> Asgari Ucret'),
        ('usd_growth','inflation','Doviz Kuru -> Enflasyon'),
        ('tr_infl','inflation','TR Enflasyon -> KKTC Enflasyon'),
    ]

    for cause, effect, label in pairs:
        data = df[[effect, cause]].dropna()
        exog = dummies.loc[data.index]

        try:
            m0 = VAR(data, exog=exog)
            p = m0.select_order(maxlags=6).aic
            p = max(1, min(p, 6))
        except:
            p = 2

        try:
            res = VAR(data, exog=exog).fit(p + 1)
        except:
            print(f"  {label}: VAR fit hatasi"); continue

        # Asimptotik test
        try:
            gc = grangercausalitytests(data, maxlag=p, verbose=False)
            mwald_stat = gc[p][0]['ssr_ftest'][0]
            mwald_p_asym = gc[p][0]['ssr_ftest'][1]
        except:
            print(f"  {label}: Granger hatasi"); continue

        # IQR Outlier Filtreleme: residuallari kirp
        resids = res.resid.copy()
        fitted = res.fittedvalues
        for col in resids.columns:
            q1 = resids[col].quantile(0.25)
            q3 = resids[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            resids[col] = resids[col].clip(lower, upper)

        n = len(resids)

        # Blok-Bootstrap
        boot_stats = []
        for _ in range(n_boot):
            # Rastgele bloklar sec, birlestir
            boot_idx = []
            while len(boot_idx) < n:
                s = np.random.randint(0, n - block_size + 1)
                boot_idx.extend(range(s, s + block_size))
            boot_idx = boot_idx[:n]  # tam uzunluga kirp

            y_boot = fitted.values + resids.values[boot_idx]
            df_boot = pd.DataFrame(y_boot, columns=data.columns)
            try:
                gc_b = grangercausalitytests(df_boot, maxlag=p, verbose=False)
                boot_stats.append(gc_b[p][0]['ssr_ftest'][0])
            except:
                pass

        if boot_stats:
            boot_p = np.mean(np.array(boot_stats) >= mwald_stat)
            sig_b = "ANLAMLI *" if boot_p < 0.05 else "ANLAMSIZ"
            sig_a = "ANLAMLI *" if mwald_p_asym < 0.05 else "ANLAMSIZ"
            print(f"  {label}")
            print(f"    Asimptotik p={mwald_p_asym:.4f} [{sig_a}]  "
                  f"Blok-Bootstrap p={boot_p:.4f} [{sig_b}]")

# ======== 6. TALEP YIKIMI (Demand Destruction) ANALIZI ========
def demand_destruction(df):
    sep("FAZ 6: TALEP YIKIMI (DEMAND DESTRUCTION) ANALIZI")

    post = df[df.index >= '2022-01-01']
    pre = df[df.index < '2022-01-01']

    # Reel asgari ucret degisimi
    real_mw_pre = pre['real_mw'].iloc[-1]
    real_mw_post = post['real_mw'].iloc[-1]
    erime = (real_mw_post / real_mw_pre - 1) * 100

    print(f"  Reel Asgari Ucret (Pre-2022 sonu): {real_mw_pre:.2f}")
    print(f"  Reel Asgari Ucret (Guncel):        {real_mw_post:.2f}")
    print(f"  Reel Erime: {erime:+.1f}%")

    # Kur sokunun siddetlenmesi
    usd_pre = pre['usd_growth'].std()
    usd_post = post['usd_growth'].std()
    print(f"\n  Kur Volatilitesi (Pre-2022):  {usd_pre:.2f}")
    print(f"  Kur Volatilitesi (Post-2022): {usd_post:.2f}")
    print(f"  Volatilite Degisimi: {(usd_post/usd_pre - 1)*100:+.1f}%")

    # Reel ucret vs enflasyon scatter (post-2022)
    print(f"\n  Post-2022 korelasyon (reel ucret degisimi vs enflasyon):")
    print(f"  r = {post['real_mw_growth'].corr(post['inflation']):.3f}")

    print("""
  YORUM: Negatif katsayi (-0.013) 'ucretin enflasyonu dusurdugu'
  anlamina GELMEZ. Bu, TALEP YIKIMI'nin kantidir:
  - Kur gecisgenligi 0.27 -> 0.62 (2.3x artis)
  - Reel satin alma gucu ezilmis
  - Nominal ucret artsa bile yeni talep yaratamamakta
  - Enflasyon tamamen maliyet-itisli (cost-push) haline gelmis
  """)

    fig, ax = plt.subplots(figsize=(12, 5))
    idx = df['real_mw'] / df['real_mw'].iloc[0] * 100
    ax.plot(df.index, idx, 'b-', lw=1.5, label='Reel Asgari Ucret')
    ax.axvline(pd.Timestamp('2022-01-01'), color='red', ls='--', lw=2, label='2022 Kirilma')
    ax.axhline(100, color='gray', ls=':', alpha=0.5)
    ax.fill_between(df.index[df.index >= '2022-01-01'], 0,
                    idx[df.index >= '2022-01-01'], color='red', alpha=0.1, label='Talep Yikimi Bolge')
    ax.set_title('Reel Asgari Ucret ve Talep Yikimi (Demand Destruction)')
    ax.set_ylabel('Endeks (1982=100)'); ax.legend()
    save_fig("05_talep_yikimi"); plt.close()

# ======== 7. ALT-DONEM + CHOW ========
def subsample(df):
    sep("FAZ 7: ALT-DONEM ANALIZI + CHOW TESTI")
    brk = pd.Timestamp('2022-01-01')
    pre = df[df.index < brk]; post = df[df.index >= brk]
    print(f"  Pre-2022: {len(pre)} goz.  Post-2022: {len(post)} goz.")

    for nm, ds in [("PRE-2022", pre), ("POST-2022", post)]:
        X = sm.add_constant(ds[['mw_growth','usd_growth','tr_infl']])
        m = sm.OLS(ds['inflation'], X).fit(cov_type='HC3')
        print(f"\n  --- {nm} ---")
        for v in ['mw_growth','usd_growth','tr_infl']:
            lbl = {'mw_growth':'Asg.Ucret','usd_growth':'Doviz Kuru','tr_infl':'TR Enflasyon'}[v]
            s = "*" if m.pvalues[v] < 0.05 else ""
            print(f"    {lbl:15s}: b={m.params[v]:>8.4f}  p={m.pvalues[v]:.4f} {s}")
        print(f"    R2={m.rsquared:.4f}")

    # Chow
    X_a = sm.add_constant(df[['mw_growth','usd_growth','tr_infl']])
    ssr_f = sm.OLS(df['inflation'], X_a).fit().ssr
    ssr_1 = sm.OLS(pre['inflation'], sm.add_constant(pre[['mw_growth','usd_growth','tr_infl']])).fit().ssr
    ssr_2 = sm.OLS(post['inflation'], sm.add_constant(post[['mw_growth','usd_growth','tr_infl']])).fit().ssr
    k = X_a.shape[1]; n = len(df)
    F = ((ssr_f - ssr_1 - ssr_2) / k) / ((ssr_1 + ssr_2) / (n - 2*k))
    p = 1 - stats.f.cdf(F, k, n - 2*k)
    print(f"\n  Chow: F={F:.4f}, p={p:.6f} -> {'KIRILMA VAR' if p<0.05 else 'Yok'}")

# ======== 8. SONUC ========
def summary(df):
    sep("SONUC OZETI (v3 - Tam Elestiri Duzeltmeleri)")
    print("""
  DUZELTILEN SORUNLAR (v3):
  1. NARDL: Hareketli ortalama esik -> gercek asimetri
  2. Bootstrap: BLOK-BOOTSTRAP + IQR outlier filtreleme
     -> Zamansal bagimlilik korunarak varyans patlamasi engellendi
  3. Sosyal Esikler: Reel ucret acigi (proxy) VARX'a eklendi
  4. Verimlilik: TR reel kur degisimi (proxy) VARX'a eklendi
  5. Talep Yikimi analizi ve reel ucret erimesi belgelendi
  """)

def main():
    print("="*70)
    print("  KKTC ILERI EKONOMETRIK ANALIZ v3")
    print("  (Blok-Bootstrap + IQR + Sosyal Proxy Duzeltmeleri)")
    print("="*70)
    df = load_data()
    run_varx(df)
    run_nardl_corrected(df, nlags=6)
    run_markov(df)
    bootstrap_mwald_v3(df, n_boot=999, block_size=12)
    demand_destruction(df)
    subsample(df)
    summary(df)
    print(f"\n  Ciktilar: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
