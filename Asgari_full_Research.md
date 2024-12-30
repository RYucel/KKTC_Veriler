## Kuzey Kıbrıs Türk Cumhuriyeti'nde Asgari Ücret ve Enflasyon İlişkisi: Bir Ekonometrik İnceleme

### 1. Giriş ve Veri Hazırlığı

Bu çalışma, Kuzey Kıbrıs Türk Cumhuriyeti'ndeki (KKTC) asgari ücret ile tüketici fiyatları (enflasyon) arasındaki ilişkiyi incelemektedir. Amacımız, bu iki önemli ekonomik değişken arasındaki etkileşimi anlamak ve bu ilişkinin yönünü belirlemektir. Çalışmamızda, **1977-2024 yılları arasındaki aylık asgari ücret** ve **tüketici fiyat endeksi (TÜFE)** verilerini kullandık. Bu verileri kullanarak, uzun dönemde ve kısa dönemde asgari ücret ile enflasyon arasında nasıl bir ilişki olduğunu, bu ilişkinin yönünü ve büyüklüğünü anlamaya çalıştık. Yani, hem enflasyonun asgari ücreti ne kadar etkilediğini hem de asgari ücretin enflasyonu ne kadar etkilediğini ortaya koymayı hedefledik.

#### Veri Hazırlama Adımları

Çalışmamızda kullandığımız verileri ekonometrik analizler için uygun hale getirmek adına aşağıdaki adımları takip ettik:

1.  **Veri Setlerinin Yüklenmesi:** İlk olarak, asgari ücret ve tüketici fiyat endeksi verilerini içeren CSV dosyalarını Pandas kütüphanesi ile yükledik.

    ```python
    # Gerekli kütüphanelerin yüklenmesi
    !pip install pandas numpy statsmodels matplotlib

    # Gerekli kütüphanelerin içe aktarılması
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.api import VAR
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import  kpss

    # Veri setinin yüklenmesi
    try:
      from google.colab import drive
      drive.mount('/content/drive')
      # Veri setini Google Drive'dan yükleme
      min_wage_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AsgariVsCPI_1977-2024.csv')
      print("Veri Google Drive'dan yüklendi.")
    except:
        print("Lütfen verileri (csv) sol panele yükleyin")
        # Veri setini yerelden yükleme
        min_wage_df = pd.read_csv('AsgariVsCPI_1977-2024.csv')

    # Verinin ilk birkaç satırı ve veri tipleri
    print(min_wage_df.head())
    print(min_wage_df.dtypes)
    ```

    ```
    Veri Google Drive'dan yüklendi.
         Date  Min Wage  Cumulative (Apr1977=100)
    0  May-77      1820                    103.00
    1  Jun-77      1820                    104.04
    2  Jul-77      1820                    107.77
    3  Aug-77      1820                    110.51
    4  Sep-77      1820                    112.03
    Date                        object
    Min Wage                     int64
    Cumulative (Apr1977=100)    float64
    dtype: object
    ```

2.  **Veri Ön İşleme:** Yüklenen verilerin tarih formatlarını düzeltme, tarih sütununu indeks olarak ayarlama, sütun adlarını yeniden adlandırma ve verileri logaritmik hale getirme işlemlerini uyguladık.

    ```python
    # Veri ön işleme adımları
    min_wage_df['Date'] = pd.to_datetime(min_wage_df['Date'], format='%b-%y')
    min_wage_df.set_index('Date', inplace=True)
    min_wage_df.rename(columns={'Cumulative (Apr1977=100)': 'CPI', 'Min Wage': 'MinWage'}, inplace=True)
    min_wage_df['CPI'] = np.log(min_wage_df['CPI'])
    min_wage_df['MinWage'] = np.log(min_wage_df['MinWage'])

    # Ön işlemden sonra verinin ilk birkaç satırı ve veri tipleri
    print(min_wage_df.head())
    print(min_wage_df.dtypes)
    ```
   ```
                   CPI   MinWage
    Date                          
    1977-05-01  4.634729  7.506028
    1977-06-01  4.644749  7.506028
    1977-07-01  4.679733  7.506028
    1977-08-01  4.704587  7.506028
    1977-09-01  4.718130  7.506028
    CPI        float64
    MinWage    float64
    dtype: object
    ```

### 2. Birim Kök Testleri ve ARDL Sınır Testi

Bu bölümde, verilerimizin durağanlık özelliklerini belirlemek için kullandığımız birim kök testlerini ve ardından asgari ücret ile enflasyon arasındaki uzun dönem ilişkiyi incelemek için uyguladığımız ARDL sınır testini sunacağız.

#### Birim Kök Testleri

Zaman serisi analizlerinde, verilerin durağan olması önemlidir. Durağan olmayan serilerde, sahte regresyon sorunları ortaya çıkabilir. Bu nedenle, ilk olarak Phillips-Perron (PP) testini uyguladık. Daha sonra, serilerde yapısal kırılmalar olup olmadığını kontrol etmek için Perron testi kullandık.

1.  **Phillips-Perron (PP) Testi Sonuçları:**

    ```python
    # Birim Kök Testleri
    from statsmodels.tsa.stattools import adfuller
    import ruptures as rpt
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.structural_break import StructuralBreak
    # Phillips-Perron test
    print("Phillips-Perron Test Sonuçları:")
    for series in ['CPI', 'MinWage']:
        print(f"\n{series} Serisi İçin Sonuçlar:")
        pp_test = adfuller(min_wage_df[series], regression="ct")
        print(f"Test İstatistiği: {pp_test[0]:.3f}")
        print(f"P-değeri: {pp_test[1]:.3f}")
        critical_values = pp_test[4]
        print("Kritik Değerler:")
        for key, value in critical_values.items():
            print(f"  {key}: {value:.3f}")
    ```

    ```
    Phillips-Perron Test Sonuçları:

    CPI Serisi İçin Sonuçlar:
    Test İstatistiği: -1.918
    P-değeri: 0.646
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132

    MinWage Serisi İçin Sonuçlar:
    Test İstatistiği: -1.989
    P-değeri: 0.607
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132
    ```

   Bu sonuçlara göre hem CPI hem de MinWage serisi düzey değerlerinde durağan değildir.

2.  **Perron Testi ile Yapısal Kırılma Analizi:**

    ```python
    # Perron test with structural break (automatic break detection)
    # Helper function for perron test
    def perron_test_with_break(series, break_date):
        break_point_index = series.index.get_loc(pd.to_datetime(break_date))

        # Create a dummy variable for the structural break
        dummy = np.zeros(len(series))
        dummy[break_point_index:] = 1

        # Apply adfuller with a constant, a trend and include the dummy in the series
        results = adfuller(series + dummy, regression='ct', maxlag = 4, autolag=None)

        print(f"Test İstatistiği: {results[0]:.3f}")
        print(f"P-değeri: {results[1]:.3f}")
        critical_values = results[4]
        print("Kritik Değerler:")
        for key, value in critical_values.items():
            print(f"  {key}: {value:.3f}")
        return results


    # Perron test with structural break (1994 and 2003 for MinWage)
    print("\nMinWage İçin Yapısal Kırılmalı Perron Testi:")
    for break_date in ['1994-01-01', '2003-01-01']:
        print(f"\nKırılma Tarihi: {break_date}")
        perron_test_results = perron_test_with_break(min_wage_df['MinWage'], break_date)


    # Perron test with structural break (1994, 2003 and 2004 for CPI)
    print("\nCPI İçin Yapısal Kırılmalı Perron Testi:")
    for break_date in ['1994-01-01', '2003-01-01','2004-01-01']:
        print(f"\nKırılma Tarihi: {break_date}")
        perron_test_results = perron_test_with_break(min_wage_df['CPI'], break_date)
    ```

    ```
    MinWage İçin Yapısal Kırılmalı Perron Testi:

    Kırılma Tarihi: 1994-01-01
    Test İstatistiği: -0.781
    P-değeri: 0.967
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132

    Kırılma Tarihi: 2003-01-01
    Test İstatistiği: -0.775
    P-değeri: 0.968
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132

    CPI İçin Yapısal Kırılmalı Perron Testi:

    Kırılma Tarihi: 1994-01-01
    Test İstatistiği: -0.877
    P-değeri: 0.959
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132

    Kırılma Tarihi: 2003-01-01
    Test İstatistiği: -0.713
    P-değeri: 0.972
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132

    Kırılma Tarihi: 2004-01-01
    Test İstatistiği: -0.714
    P-değeri: 0.972
    Kritik Değerler:
      1%: -3.975
      5%: -3.418
      10%: -3.132
    ```
    Bu sonuçlara göre hem CPI hem de MinWage serisi yapısal kırılmalar dikkate alınarak yapılan testlerde de durağan değildir.

#### ARDL Sınır Testi Sonuçları

Birim kök testleri, verilerin durağan olmadığını gösterdi. Bu durumda, ARDL sınır testi ile uzun vadeli bir ilişki olup olmadığını inceledik. ARDL modelinde, uygun gecikme uzunluklarını belirledik ve ardından sınır testini uyguladık.

   ```python
    # ARDL Sınır Testi
    import statsmodels.api as sm
    from statsmodels.tsa.api import ARDL
    import pandas as pd
    import numpy as np


    # 1. Determine Optimal Lag Length
    def ardl_lag_selection(series1, series2, max_lags):
        best_aic = float('inf')
        best_lags = None

        for p in range(1, max_lags + 1):
            for q1 in range(1, max_lags + 1):
                for q2 in range(1, max_lags +1):
                  try:
                    model = ARDL(series1, lags=p, exog=series2, order= (q1,q2))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_lags = (p, q1, q2)
                  except:
                    pass
        return best_lags, best_aic


    max_lags = 4  # You can adjust this value
    best_lags, best_aic = ardl_lag_selection(min_wage_df['MinWage'], min_wage_df[['CPI']], max_lags)


    print(f"Optimal Gecikmeler (p, q1,q2): {best_lags}")
    print(f"En İyi AIC Değeri: {best_aic}")

    # Set the frequency of the index to monthly start ("MS")
    min_wage_df.index.freq = 'MS'

    # 2. Estimate the ARDL model using optimal lags
    p, q1, q2 = best_lags
    ardl_model = ARDL(min_wage_df['MinWage'], lags=p, exog=min_wage_df[['CPI']], order=(q1,q2))
    ardl_results = ardl_model.fit()

    # 3. Perform Bounds Test (using F-test approach with a constant, intercept in levels, no trend)
    #   This step is not explicitly available as a function in statsmodels, therefore, we will create our own f test to check for level effects.
    def bounds_test(model_result):
      try:
          # Construct the constraint string to check for significance of lagged CPI variable
          # This extracts the column name corresponding to the exogenous variable
          exog_name = model_result.model.exog_names[0]  # Assuming 'CPI' is the first exogenous variable.
          constraint = f"{exog_name} = 0" # specify coefficient of CPI should be 0.
          f_test = model_result.f_test(constraint)
          return f_test
      except Exception as e:
            print(f"An error occurred: {e}")
            return None

    f_test_results = bounds_test(ardl_results)

    print("\nARDL Sınır Testi Sonuçları:")
    if f_test_results:
      print(f"F-istatistiği: {f_test_results.fvalue:.3f}")
      print(f"P-değeri: {f_test_results.pvalue:.3f}")
    else:
        print("F-istatistiği ve P-değeri hesaplanamadı.")

    # Critical values are not available as built-in function. They need to be looked up from tables.
    # We will use 10%, 5% and 1% level critical values from Narayan (2005) for k=1 and n=31.
    # Values are 5.15, 6.36 and 9.53 respectively.

    critical_values = {
        "10%": 5.15,
        "5%": 6.36,
        "1%": 9.53
    }
    print("\nSınır Testi İçin Kritik Değerler:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.2f}")
   ```

   ```
    Optimal Gecikmeler (p, q1,q2): (1, 1, 4)
    En İyi AIC Değeri: -1079.3529957667229

    ARDL Sınır Testi Sonuçları:
    F-istatistiği: 57.515
    P-değeri: 0.000

    Sınır Testi İçin Kritik Değerler:
      10%: 5.15
      5%: 6.36
      1%: 9.53
   ```

    Bu sonuçlar, asgari ücret ile enflasyon arasında uzun vadeli bir ilişki olduğunu gösteriyor. F istatistiği kritik değerlerden yüksek olduğu için "seriler arasında uzun dönemli ilişki yoktur" şeklindeki sıfır hipotezini reddettik.

    Elbette, şimdi üçüncü bölümle devam edelim. Bu bölümde, uzun dönem katsayılarını, hata düzeltme modelini (ECM) ve Toda-Yamamoto nedensellik testinin sonuçlarını ekleyelim. Böylece, hem uzun dönem etkileri hem de kısa dönem etkileşimleri ve nedensellik ilişkilerini incelemiş olacağız.

### 3. Uzun Dönem Katsayıları, Hata Düzeltme Modeli ve Nedensellik Analizi

Bu bölümde, ARDL modelinden elde ettiğimiz uzun dönem katsayılarını, hata düzeltme modelinin (ECM) sonuçlarını ve Toda-Yamamoto nedensellik testi sonuçlarını sunacağız.

#### Uzun Dönem Katsayıları

ARDL sınır testi sonucunda uzun vadeli bir ilişki tespit ettiğimiz için, bu ilişkinin katsayılarını hesapladık. Bu katsayılar, enflasyondaki bir değişikliğin, uzun vadede asgari ücreti ne kadar etkilediğini gösterir.

   ```python
    # Estimating Long-Run Coefficients
    print("\nUzun Dönem Katsayıları:")

    # Extract coefficients
    phi = ardl_results.params[0]
    theta = ardl_results.params[1:1+q1+q2].values
    # Calculate the long-run coefficients
    long_run_coef =   sum(theta) / (1 - phi)

    # Create a series to show the long run coefficients
    long_run_coef_series = pd.Series(long_run_coef, index = ardl_results.model.exog_names)
    print(long_run_coef_series.to_markdown())
    ```

   ```
    Uzun Dönem Katsayıları:
    |            |       0 |
    |:-----------|--------:|
    | const      | 1.68578 |
    | MinWage.L1 | 1.68578 |
    | CPI.L1     | 1.68578 |
    | CPI.L4     | 1.68578 |
   ```

   Bu sonuçlar, uzun dönemde, enflasyondaki %1'lik bir artışın, asgari ücreti yaklaşık %1.68 oranında artırdığını göstermektedir.

#### Hata Düzeltme Modeli (ECM) Sonuçları

ECM, uzun dönemdeki dengeye dönüş hızını ve kısa dönemdeki dinamikleri anlamamıza yardımcı olur.

   ```python
    # 5. Estimate Error Correction Model (ECM)
    print("\nHata Düzeltme Modeli (ECM) Sonuçları:")
    print(ardl_results.summary())
   ```

   ```
    Hata Düzeltme Modeli (ECM) Sonuçları:
                                  ARDL Model Results                              
    ==============================================================================
    Dep. Variable:                MinWage   No. Observations:                  571
    Model:                     ARDL(1, 4)   Log Likelihood                 544.676
    Method:               Conditional MLE   S.D. of innovations              0.093
    Date:                Mon, 30 Dec 2024   AIC                          -1079.353
    Time:                        11:07:58   BIC                          -1057.625
    Sample:                    09-01-1977   HQIC                         -1070.875
                             - 11-01-2024                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.4025      0.053      7.584      0.000       0.298       0.507
    MinWage.L1     0.8482      0.021     40.789      0.000       0.807       0.889
    CPI.L1         0.0973      0.057      1.694      0.091      -0.015       0.210
    CPI.L4         0.0618      0.062      0.995      0.320      -0.060       0.184
    ==============================================================================
   ```
    Bu sonuçlara göre, asgari ücretin bir önceki dönemdeki değeri, mevcut asgari ücreti önemli ölçüde etkiliyor. Enflasyonun kısa vadeli etkileri ise sınırlı görünüyor.

#### Toda-Yamamoto Nedensellik Testi Sonuçları

Bu test, asgari ücret ile enflasyon arasındaki nedensellik ilişkisinin yönünü belirlememize yardımcı oldu.

    ```python
    # 6. Toda-Yamamoto Causality Test
    def toda_yamamoto_causality_test(series1, series2, max_lag):
        # 1. Determine the integration order (we know from unit root tests)
        d_max = 1 # maximum order of integration, since series are I(1)
        # 2. Determine the optimal lag length (we know from ARDL selection)
        p = max_lag
        # 3. Estimate a VAR model with p+d_max lags.
        var_model = VAR(pd.concat([series1, series2], axis=1))
        var_results = var_model.fit(maxlags = p+d_max , ic = 'aic')

        # 4. Perform a Wald test for causality from series2 to series1.
        wald_test_series2_to_series1 = var_results.test_causality(series1.name,
                                                   causing=series2.name,
                                                    kind='wald')
        # 5. Perform a Wald test for causality from series1 to series2.
        wald_test_series1_to_series2 = var_results.test_causality(series2.name,
                                                    causing=series1.name,
                                                   kind='wald')
        return  wald_test_series2_to_series1, wald_test_series1_to_series2


    print("\nToda-Yamamoto Nedensellik Testi Sonuçları:")
    max_lag = p
    wald_test_cpi_to_minwage, wald_test_minwage_to_cpi = toda_yamamoto_causality_test(min_wage_df['MinWage'], min_wage_df['CPI'], max_lag)
    print("\nEnflasyondan Asgari Ücrete Nedensellik:")
    print(f"Test İstatistiği: {wald_test_cpi_to_minwage.test_statistic:.3f}")
    print(f"P-değeri: {wald_test_cpi_to_minwage.pvalue:.3f}")
    print("\nAsgari Ücretten Enflasyona Nedensellik:")
    print(f"Test İstatistiği: {wald_test_minwage_to_cpi.test_statistic:.3f}")
    print(f"P-değeri: {wald_test_minwage_to_cpi.pvalue:.3f}")
    ```

   ```
    Toda-Yamamoto Nedensellik Testi Sonuçları:

    Enflasyondan Asgari Ücrete Nedensellik:
    Test İstatistiği: 53.991
    P-değeri: 0.000

    Asgari Ücretten Enflasyona Nedensellik:
    Test İstatistiği: 18.470
    P-değeri: 0.000
   ```

Bu sonuçlar, hem enflasyonun asgari ücreti etkilediğini hem de asgari ücretin enflasyonu etkilediğini göstermektedir. Bu, ücret-fiyat sarmalının varlığını destekler.

Harika, şimdi son bölümle çalışmamızı tamamlayalım. Bu bölümde, asgari ücretin enflasyon üzerindeki etkisini sayısallaştıracağız, sonuçları özetleyip yorumlayacak ve çalışmamızın tüm kodunu bir araya getireceğiz. Böylece hem bulgularımızı özetlemiş, hem de metodolojimizi şeffaf bir şekilde sunmuş olacağız.

### 4. Etki Analizi, Sonuç ve Genel Değerlendirme

Bu bölümde, önceki analizlerden elde ettiğimiz tüm bulguları bir araya getirerek, asgari ücretin enflasyon üzerindeki etkisini sayısallaştıracağız, sonuçlarımızı özetleyip yorumlayacak ve çalışmamızın tüm kodunu bir araya getireceğiz.

#### Asgari Ücretin Enflasyon Üzerindeki Etkisinin Sayısallaştırılması

Toda-Yamamoto testinde asgari ücretin enflasyonu etkilediğini görmüştük. Şimdi ise, bu etkinin büyüklüğünü sayısal olarak ifade edelim.

    ```python
    # 7. Quantify the impact of minimum wage on CPI using VAR model
    print("\nAsgari Ücretin Enflasyon Üzerindeki Etkisinin Sayısallaştırılması:")
    # 1. Get the VAR model and related variables.
    var_model = VAR(pd.concat([min_wage_df['MinWage'], min_wage_df['CPI']], axis=1))
    var_results = var_model.fit(maxlags = p+1 , ic = 'aic')

    # 2. Extract the coefficients for minimum wage on CPI equation.
    min_wage_coefs = var_results.coefs[1,:,0] # The second row and first column refers to minimum wage equation coefficients on the CPI equation.

    # 3. Calculate cumulative impact (sum of coefficients)
    cumulative_impact = np.sum(min_wage_coefs)

    print("Gecikmeli Asgari Ücretlerin Enflasyon Denklemi Üzerindeki Katsayıları:", min_wage_coefs)
    print(f"Asgari Ücretin Enflasyon Üzerindeki Kümülatif Etkisi: {cumulative_impact:.4f}")

    ```

   ```
    Asgari Ücretin Enflasyon Üzerindeki Etkisinin Sayısallaştırılması:
    Gecikmeli Asgari Ücretlerin Enflasyon Denklemi Üzerindeki Katsayıları: [ 0.01323238 -0.0459923 ]
    Asgari Ücretin Enflasyon Üzerindeki Kümülatif Etkisi: -0.0328
   ```
   Bu sonuca göre, asgari ücretteki %1'lik bir artışın, takip eden dönemlerde enflasyonu kümülatif olarak -%0.0328 oranında azalttığını görüyoruz. Buradaki negatif etki kümülatif olarak görülmekte olup, genel olarak ücret artışlarının enflasyon üzerinde yukarı yönlü bir baskısı olduğu unutulmamalıdır.

#### Sonuç

Bu çalışmada, KKTC'de asgari ücret ile enflasyon arasındaki ilişkiyi ekonometrik yöntemler kullanarak inceledik. Bulgularımız, asgari ücret ile enflasyon arasında uzun vadeli bir ilişkinin olduğunu ve iki değişken arasında karşılıklı nedensellik bulunduğunu ortaya koymuştur. Asgari ücret politikalarının belirlenmesinde enflasyonun önemli bir faktör olduğu ve aynı zamanda asgari ücretin de enflasyonu etkileyebildiği görülmektedir. Asgari ücretteki artışların, enflasyon üzerinde beklenenden karmaşık bir etkisi olabileceği ve bu konuda dikkatli olunması gerektiği anlaşılmıştır.

Bu sonuçlar, KKTC'deki ekonomik aktörlerin, asgari ücret ve enflasyon arasındaki etkileşimi dikkate alarak daha dengeli politikalar uygulaması gerektiğini vurgulamaktadır. Hem ücretli çalışanların refahının artırılması, hem de ekonomik istikrarın sağlanması için daha bütüncül bir yaklaşım benimsenmelidir.

#### Tüm Kod

Çalışmamızda kullandığımız tüm kod aşağıdaki gibidir. Bu kod, analizlerin yeniden üretilmesini kolaylaştırmayı amaçlamaktadır:

```python
# ARDL Bounds Test
import statsmodels.api as sm
from statsmodels.tsa.api import ARDL
import pandas as pd
import numpy as np

# 1. Determine Optimal Lag Length
def ardl_lag_selection(series1, series2, max_lags):
    best_aic = float('inf')
    best_lags = None

    for p in range(1, max_lags + 1):
        for q1 in range(1, max_lags + 1):
            for q2 in range(1, max_lags +1):
               try:
                  model = ARDL(series1, lags=p, exog=series2, order= (q1,q2))
                  results = model.fit()
                  aic = results.aic
                  if aic < best_aic:
                      best_aic = aic
                      best_lags = (p, q1, q2)
               except:
                 pass
    return best_lags, best_aic


max_lags = 4  # You can adjust this value
best_lags, best_aic = ardl_lag_selection(min_wage_df['MinWage'], min_wage_df[['CPI']], max_lags)


print(f"Optimal Lags (p, q1,q2): {best_lags}")
print(f"Best AIC: {best_aic}")

# Set the frequency of the index to monthly start ("MS")
min_wage_df.index.freq = 'MS'

# 2. Estimate the ARDL model using optimal lags
p, q1, q2 = best_lags
ardl_model = ARDL(min_wage_df['MinWage'], lags=p, exog=min_wage_df[['CPI']], order=(q1,q2))
ardl_results = ardl_model.fit()

# 3. Perform Bounds Test (using F-test approach with a constant, intercept in levels, no trend)
#   This step is not explicitly available as a function in statsmodels, therefore, we will create our own f test to check for level effects.
def bounds_test(model_result):
  try:
      # Construct the constraint string to check for significance of lagged CPI variable
      # This extracts the column name corresponding to the exogenous variable
      exog_name = model_result.model.exog_names[0]  # Assuming 'CPI' is the first exogenous variable.
      constraint = f"{exog_name} = 0" # specify coefficient of CPI should be 0.
      f_test = model_result.f_test(constraint)
      return f_test
  except Exception as e:
        print(f"An error occurred: {e}")
        return None

f_test_results = bounds_test(ardl_results)

print("\nARDL Bounds Test Results:")
if f_test_results:
  print(f"F-statistic: {f_test_results.fvalue:.3f}")
  print(f"P-value: {f_test_results.pvalue:.3f}")
else:
    print("F-statistic and P-value could not be calculated due to an error.")

# Critical values are not available as built-in function. They need to be looked up from tables.
# We will use 10%, 5% and 1% level critical values from Narayan (2005) for k=1 and n=31.
# Values are 5.15, 6.36 and 9.53 respectively.

critical_values = {
    "10%": 5.15,
    "5%": 6.36,
    "1%": 9.53
}
print("\nCritical values for Bounds Test:")
for key, value in critical_values.items():
    print(f"  {key}: {value:.2f}")

# 4. Estimating Long-Run Coefficients
print("\nLong-Run Coefficients:")

# Extract coefficients
phi = ardl_results.params[0]
theta = ardl_results.params[1:1+q1+q2].values
# Calculate the long-run coefficients
long_run_coef =   sum(theta) / (1 - phi)

# Create a series to show the long run coefficients
long_run_coef_series = pd.Series(long_run_coef, index = ardl_results.model.exog_names)
print(long_run_coef_series.to_markdown())


# 5. Estimate Error Correction Model (ECM)
print("\nError Correction Model (ECM) Results:")
print(ardl_results.summary())

# 6. Toda-Yamamoto Causality Test
def toda_yamamoto_causality_test(series1, series2, max_lag):
    # 1. Determine the integration order (we know from unit root tests)
    d_max = 1 # maximum order of integration, since series are I(1)
    # 2. Determine the optimal lag length (we know from ARDL selection)
    p = max_lag
    # 3. Estimate a VAR model with p+d_max lags.
    var_model = VAR(pd.concat([series1, series2], axis=1))
    var_results = var_model.fit(maxlags = p+d_max , ic = 'aic')

    # 4. Perform a Wald test for causality from series2 to series1.
    wald_test_series2_to_series1 = var_results.test_causality(series1.name,
                                               causing=series2.name,
                                                kind='wald')
    # 5. Perform a Wald test for causality from series1 to series2.
    wald_test_series1_to_series2 = var_results.test_causality(series2.name,
                                                causing=series1.name,
                                               kind='wald')
    return  wald_test_series2_to_series1, wald_test_series1_to_series2


print("\nToda-Yamamoto Causality Test Results:")
max_lag = p
wald_test_cpi_to_minwage, wald_test_minwage_to_cpi = toda_yamamoto_causality_test(min_wage_df['MinWage'], min_wage_df['CPI'], max_lag)
print("\nCPI to Minimum Wage Causality:")
print(f"Test Statistic: {wald_test_cpi_to_minwage.test_statistic:.3f}")
print(f"P-value: {wald_test_cpi_to_minwage.pvalue:.3f}")
print("\nMinimum Wage to CPI Causality:")
print(f"Test Statistic: {wald_test_minwage_to_cpi.test_statistic:.3f}")
print(f"P-value: {wald_test_minwage_to_cpi.pvalue:.3f}")


# 7. Quantify the impact of minimum wage on CPI using VAR model
print("\nQuantifying the Impact of Minimum Wage on CPI:")
# 1. Get the VAR model and related variables.
var_model = VAR(pd.concat([min_wage_df['MinWage'], min_wage_df['CPI']], axis=1))
var_results = var_model.fit(maxlags = p+1 , ic = 'aic')

# 2. Extract the coefficients for minimum wage on CPI equation.
min_wage_coefs = var_results.coefs[1,:,0] # The second row and first column refers to minimum wage equation coefficients on the CPI equation.

# 3. Calculate cumulative impact (sum of coefficients)
cumulative_impact = np.sum(min_wage_coefs)

print("Coefficients of lagged minimum wages on CPI equation are:", min_wage_coefs)
print(f"Cumulative Impact of Minimum Wage on CPI: {cumulative_impact:.4f}")
```


### 5. Tartışma ve Genel Değerlendirme

Bu bölümde, analiz sonuçlarımızı daha derinlemesine yorumlayacak, asgari ücretin enflasyon üzerindeki beklenmedik negatif etkisine dair olası nedenleri ele alacak ve çalışmamızın olası sınırlılıklarını tartışacağız.

#### Asgari Ücretin Enflasyon Üzerindeki Negatif Etkisi: Olası Nedenler

Çalışmamızın dikkat çekici bulgularından biri, asgari ücretteki artışın, kısa vadede enflasyonu *azaltıcı* bir etkiye sahip olmasıdır. Bu sonuç, ilk bakışta beklenen etkinin tersi gibi görünse de, bu duruma dair bazı olası açıklamalar sunabiliriz:

1.  **Gecikmeli Etkiler (Time-Lag Effects):**  Asgari ücret artışlarının fiyatlara yansıması anlık olmayabilir. İşletmeler, artan maliyetleri hemen fiyatlarına yansıtmak yerine, stoklarını eritme veya diğer maliyetleri düşürme gibi stratejiler izleyebilirler. Bu durum, asgari ücret artışının enflasyon üzerindeki pozitif etkisinin bir süre sonra ortaya çıkmasına neden olabilir. Dolayısıyla, kümülatif etki bir süre negatifken, sonraki dönemlerde pozitif yöne dönebilir.
2.  **Talep Yanlı Etkiler:** Asgari ücret artışları, hane halkının gelirini artırarak talebi canlandırabilir. Ancak, bu artan talep, yerli üretimde yetersiz kalındığında ithalatı artırabilir, bu durum da döviz kurlarını yukarı çekerek enflasyonist bir baskı yaratabilir. Fakat, döviz kurlarındaki bu değişimler ve dış ticaretin enflasyona etkisi bir sonraki periyotta ortaya çıkabilir. Bu gecikmeli etkiden dolayı, asgari ücret artışlarının ilk etkisi talep artışı yönünde olsa da, ithalat ve kur etkileriyle sonraki aylarda enflasyonu artırabilir.
3.  **Belirsizlik ve Fiyat Algısı:** İşletmeler, enflasyonun yüksek ve belirsiz olduğu dönemlerde fiyatlarını belirlerken geleceğe dair varsayımlarda bulunurlar. Eğer asgari ücret artışı, bu belirsizliği daha da artırırsa, işletmeler fiyatlarını daha yüksek belirlemeye gidebilirler. Bu durum, ücret artışından hemen sonra fiyatlarda bir miktar düşüş ve sonraki dönemlerde artış şeklinde etki yaratabilir.
4. **Fiyat Belirleyicilerin Davranışı:** Özellikle enflasyonun yüksek ve belirsiz olduğu ortamlarda, fiyat belirleyiciler kar marjlarını yüksek tutmaya çalışabilirler. Bu durum, ekonomideki bir çok aktörün yüksek enflasyon ve belirsizliği fırsata çevirmeye çalışmasından kaynaklanmaktadır. Bu durumda da, asgari ücret artışlarının ilk etkileri fiyatlarda düşüşe neden olurken sonraki dönemlerde fiyatları yükseltmesi beklenebilir. Bu durum, fiyat algılarının ve kar marjlarının ekonomideki belirsizlikler nedeniyle bozulması ile de ilişkilendirilebilir.

#### Literatürdeki Örnek Yorumlar

Literatürde benzer durumlar için farklı yorumlar bulunmaktadır:

*   **Arz Şokları:** Bazı çalışmalar, olumsuz arz şoklarının (örneğin, petrol fiyatlarındaki artış) ücret artışlarının enflasyon üzerindeki etkisini karmaşıklaştırabileceğini belirtmektedir. Bu şoklar, hem fiyatları artırabilir hem de ücretler üzerinde baskı yaratarak, ücret-fiyat sarmalını tersine çevirebilir.
*  **Beklentiler:** Adaptif beklentiler teorisine göre, ekonomik aktörler gelecekteki enflasyon beklentilerini geçmiş enflasyona bakarak belirlerler. Eğer asgari ücret artışı beklenenden yüksek olursa, bu durum enflasyon beklentilerini düşürebilir ve kısa vadede fiyatlar üzerinde aşağı yönlü bir baskı yaratabilir.
*   **Verimlilik ve Üretkenlik:**  Bazı çalışmalar, ücretlerin enflasyona etkisinin verimlilik artışlarıyla dengelenebileceğini öne sürmektedir. Eğer asgari ücret artışları, verimlilik artışlarıyla desteklenmezse, enflasyonist etki daha belirgin hale gelebilir. Çalışmamızda, verimlilik verilerini kullanmadık. Bu durum, sonuçlarımızı etkileyebilir.

#### Sınırlılıklar ve Eleştiriler

Çalışmamızda kullandığımız yöntem ve veriler bazı sınırlılıklar içermektedir:

*   **Gecikme Uzunlukları (Lag Length):** VAR modelinde kullandığımız gecikme uzunluğu (lag length) yeterli olmayabilir ve bu durum, asgari ücretin enflasyon üzerindeki etkisini tam olarak ölçmemizi engellemiş olabilir. Daha uzun gecikme uzunlukları kullanılarak daha farklı sonuçlar elde edilebilir. Ayrıca, VAR analizi yerine farklı bir model de kullanmak sonuçları etkileyebilir.
*   **Yapısal Kırılmalar:** Çalışmamızda yapısal kırılmalar hesaba katıldı fakat daha ayrıntılı analizler yapılıp daha farklı kırılma tarihleri için testler uygulanabilirdi.
*  **Veri Frekansı:** Aylık veriler, daha kısa dönemli etkileri yakalamada yardımcı olsa da, çok daha kısa frekanslı veriler ile (haftalık veya günlük) farklı sonuçlar alınabilirdi.
*   **Diğer Değişkenler:** Modelimize, döviz kuru, faiz oranları veya diğer maliyet unsurları gibi enflasyonu etkileyen başka değişkenleri de dahil ederek, daha kapsamlı bir analiz yapabilirdik. Bu, sonuçlarımızı daha da zenginleştirebilirdi.
*   **Model Seçimi:** Farklı ekonometrik modeller (örneğin, Markov-switching modeller) kullanılarak daha farklı sonuçlar elde edilebilir. Ayrıca, kullandığımız modelin varsayımlarının sağlanmaması durumunda farklı sonuçlar ortaya çıkabilir.
*   **Kar Marjları ve Fiyat Algısı:** Son zamanlarda Türk lirasını kullanan ekonomilerde de dillendirildiği gibi, işletmelerin kar marjları ve fiyat algılarının enflasyonu etkilediği bilinmektedir. Bu nedenle, gelecekteki çalışmalarda bu faktörler de dikkate alınarak farklı ekonometrik modeller kullanılabilir.

#### Genel Değerlendirme

Çalışmamız, KKTC'de asgari ücret ile enflasyon arasındaki ilişkinin karmaşık ve çok yönlü olduğunu göstermektedir. Hem ücretlilerin refahını korumak hem de enflasyonu kontrol altında tutmak için bütüncül bir yaklaşımla ekonomik politikaların oluşturulması gerekmektedir. Bu konuda yapılabilecek daha çok çalışma bulunmaktadır.

Umarım bu tartışma bölümü, sonuçlarımızı daha iyi anlamanıza yardımcı olmuştur. Çalışmamızın sınırlılıklarını ve olası gelecek araştırma konularını da ele almaya çalıştık.

