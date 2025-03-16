import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer

st.title("Boston Housing Dataset - Veri Analizi ve Raporlama")

# Sekmeleri oluşturma
step1, step2, step3 = st.tabs(["Veri Temizleme", "Analiz & Görselleştirme", "Raporlama & İçgörüler"])

### 1. Veri Temizleme
with step1:
    st.header("Veri Temizleme")

    @st.cache_data
    def load_data():
        """CSV dosyasını yükler."""
        df = pd.read_csv("HousingData.csv")
        return df
    
    df = load_data()

    def fill_missing_values(df):
        """Eksik verileri doldurur."""
        imputer_mean = SimpleImputer(strategy="mean")
        imputer_median = SimpleImputer(strategy="median")
        imputer_knn = KNNImputer(n_neighbors=5)

        df['CRIM'] = imputer_mean.fit_transform(df[['CRIM']])
        df['ZN'] = imputer_median.fit_transform(df[['ZN']])
        df['CHAS'] = imputer_knn.fit_transform(df[['CHAS']])
        df['CHAS'] = df['CHAS'].round().astype(int)  # KNN sonuçlarını tam sayıya çevir
        df['DIS'] = imputer_knn.fit_transform(df[['DIS']])
        df['INDUS'] = imputer_median.fit_transform(df[['INDUS']])
        df['AGE'] = imputer_mean.fit_transform(df[['AGE']])
        df['LSTAT'] = imputer_knn.fit_transform(df[['LSTAT']])
        
        return df

    df = fill_missing_values(df)

    st.subheader("Temizlenmiş Veri Kümesi")
    st.write(df.head())

    st.subheader("Eksik Veri Kontrolü")
    st.write(df.isnull().sum())

    # Veriyi CSV formatına çevirme
    cleaned_csv = df.to_csv(index=False).encode('utf-8')

    # Kullanıcıya indirme butonu ekleme
    st.download_button(
        label="Temizlenmiş Veriyi İndir",
        data=cleaned_csv,
        file_name="Boston_Housing_Cleaned.csv",
        mime="text/csv"
    )

### 2. Analiz & Görselleştirme
with step2:
    st.header("Analiz & Görselleştirme")

    def plot_graphs():
        """Çeşitli grafikler oluşturur ve Streamlit'e ekler."""

        # Bina Yaşı - Ev Fiyatı İlişkisi
        st.subheader("Bina Yaşı ile Ev Fiyatı İlişkisi")
        fig1, ax1 = plt.subplots()
        ax1.scatter(df['AGE'], df['MEDV'], alpha=0.5, color='blue')
        ax1.set_xlabel("Bina Yaşı (AGE)")
        ax1.set_ylabel("Medyan Ev Fiyatı (MEDV)")
        ax1.set_title("Bina Yaşı ile Ev Fiyatı İlişkisi")
        st.pyplot(fig1)

        # Oda Sayısı - Ev Fiyatı İlişkisi
        st.subheader("Oda Sayısı ile Ev Fiyatı İlişkisi")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['RM'], df['MEDV'], alpha=0.5, color='red')
        ax2.set_xlabel("Ortalama Oda Sayısı (RM)")
        ax2.set_ylabel("Medyan Ev Fiyatı (MEDV)")
        ax2.set_title("Oda Sayısı ile Ev Fiyatı İlişkisi")
        st.pyplot(fig2)

        # Suç Oranı - Ev Fiyatı İlişkisi
        st.subheader("Suç Oranı ile Ev Fiyatı İlişkisi")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df['CRIM'], df['MEDV'], alpha=0.5, color='purple')
        ax3.set_xlabel("Suç Oranı (CRIM)")
        ax3.set_ylabel("Medyan Ev Fiyatı (MEDV)")
        ax3.set_title("Suç Oranı ile Ev Fiyatı İlişkisi")
        st.pyplot(fig3)

        # Öğrenci-Öğretmen Oranı - Ev Fiyatı İlişkisi
        st.subheader("Öğrenci-Öğretmen Oranı ile Ev Fiyatı İlişkisi")
        fig4, ax4 = plt.subplots()
        ax4.scatter(df['PTRATIO'], df['MEDV'], alpha=0.5, color='green')
        ax4.set_xlabel("Öğrenci-Öğretmen Oranı (PTRATIO)")
        ax4.set_ylabel("Medyan Ev Fiyatı (MEDV)")
        ax4.set_title("Öğrenci-Öğretmen Oranı ile Ev Fiyatı İlişkisi")
        st.pyplot(fig4)

        # Korelasyon Isı Haritası
        st.subheader("Korelasyon Isı Haritası")
        fig5, ax5 = plt.subplots(figsize=(12,8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
        ax5.set_title("Değişkenler Arasındaki Korelasyon")
        st.pyplot(fig5)

    plot_graphs()

### 3. Raporlama & İçgörüler
with step3:
    st.header("Raporlama & İçgörüler")

    st.markdown("### Genel Bulgular:")
    st.markdown("- Oda sayısı arttıkça ev fiyatları yükseliyor.")
    st.markdown("- Suç oranı yüksek olan bölgelerde ev fiyatları düşüyor.")
    st.markdown("- Nehre yakınlık (CHAS = 1) olan evlerin fiyatları genellikle daha yüksek.")

    st.markdown("### Öneriler:")
    st.markdown("- Yüksek suç oranına sahip bölgelerde yatırım yaparken dikkatli olunmalı.")
    st.markdown("- Eğitim kalitesi yüksek bölgeler (PTRATIO düşük) yatırım için avantajlı olabilir.")
    st.markdown("- Oda sayısı fazla olan evler genellikle daha değerli.")
