import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Menambahkan judul utama untuk dashboard
st.title("Dashboard Bike Sharing")

# Menambahkan deskripsi singkat tentang dataset dan metode analisis
st.markdown("""
### Tentang Dataset
Dataset ini berisi data penyewaan sepeda yang dikumpulkan dari sistem bike sharing di dua skala waktu: **harian** dan **per jam**. Informasi yang tersedia mencakup:
- Kondisi cuaca, suhu, kelembapan, dan kecepatan angin.
- Jumlah penyewaan sepeda yang dilakukan oleh pengguna terdaftar (**registered**) dan pengguna biasa (**casual**).

### Metode Analisis
Beberapa metode analisis digunakan dalam dashboard ini untuk mengeksplorasi pola penggunaan sepeda:
- **Exploratory Data Analysis (EDA)**: Menggambarkan pola penggunaan sepeda berdasarkan waktu, cuaca, dan kategori pengguna.
- **Clustering**: Metode **K-Means** digunakan untuk mengelompokkan hari-hari berdasarkan kondisi cuaca dan jumlah penyewaan sepeda.

Berikut adalah hasil dari analisis data penyewaan sepeda:
""")

# Memuat dataset
@st.cache
def load_data():
    day_df = pd.read_csv('day.csv')
    hour_df = pd.read_csv('hour.csv')
    return day_df, hour_df

# Memuat data
day_df, hour_df = load_data()

# Konversi kolom 'dteday' menjadi datetime
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# Menampilkan beberapa baris pertama dari kedua dataset
st.write("Dataset Harian (day.csv):")
st.write(day_df.head())

st.write("Dataset Per Jam (hour.csv):")
st.write(hour_df.head())

# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis (EDA)")

# Pola Penggunaan Sepeda Harian
st.subheader("Total Daily Bike Rentals Over Time")
plt.figure(figsize=(14, 7))
plt.plot(day_df['dteday'], day_df['cnt'], label='Total Rentals', color='blue')
plt.title('Total Daily Bike Rentals Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Rentals')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# Insight for Daily Rentals
st.markdown("""
**Insight:**
- Pola penyewaan sepeda harian menunjukkan tren musiman yang jelas.
- Puncak penggunaan sepeda terjadi pada musim panas, dan jumlah penyewaan menurun signifikan pada musim dingin.
""")

# Distribusi Penyewaan Sepeda Per Jam
st.subheader("Hourly Bike Rentals Distribution")
plt.figure(figsize=(12, 6))
sns.boxplot(x='hr', y='cnt', data=hour_df)
plt.title('Hourly Bike Rentals Distribution')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rentals')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# Insight for Hourly Rentals
st.markdown("""
**Insight:**
- Penyewaan sepeda paling tinggi terjadi pada jam 8 pagi dan 5-6 sore, yang mencerminkan waktu komuter.
- Penggunaan sepeda sangat rendah pada malam hari.
""")

# Korelasi antara Faktor Cuaca dan Penyewaan Sepeda
st.subheader("Correlation Between Weather Factors and Bike Rentals (Daily)")
plt.figure(figsize=(10, 6))
sns.heatmap(day_df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Weather Factors and Bike Rentals (Daily)')
plt.tight_layout()
st.pyplot(plt)

# Insight for Weather Correlation
st.markdown("""
**Insight:**
- Temperatur memiliki korelasi positif yang kuat dengan jumlah penyewaan sepeda.
- Kelembapan dan kecepatan angin memiliki dampak negatif kecil, yang menandakan bahwa kondisi cuaca yang buruk sedikit menurunkan jumlah penyewaan.
""")

# Distribusi Penyewaan Berdasarkan Kondisi Cuaca
st.subheader("Bike Rentals Distribution by Weather Condition (Hourly)")
plt.figure(figsize=(10, 6))
sns.boxplot(x='weathersit', y='cnt', data=hour_df)
plt.title('Bike Rentals Distribution by Weather Condition (Hourly)')
plt.xlabel('Weather Situation (1 = Clear, 2 = Mist, 3 = Light Snow/Rain)')
plt.ylabel('Number of Rentals')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# Insight for Weather Condition
st.markdown("""
**Insight:**
- Penyewaan sepeda tertinggi terjadi pada kondisi cuaca cerah.
- Kondisi cuaca buruk seperti hujan atau kabut secara signifikan mengurangi jumlah penyewaan sepeda.
""")

# Pengguna Casual vs Registered
st.subheader("Casual vs Registered Users by Hour")
plt.figure(figsize=(12, 6))
sns.lineplot(x='hr', y='casual', data=hour_df, label='Casual Users', color='orange')
sns.lineplot(x='hr', y='registered', data=hour_df, label='Registered Users', color='blue')
plt.title('Casual vs Registered Users by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Users')
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# Insight for Casual vs Registered
st.markdown("""
**Insight:**
- Pengguna casual cenderung menyewa sepeda lebih banyak di siang hari.
- Pengguna registered lebih aktif di pagi dan sore hari saat waktu komuter.
""")

# Visualisasi Pola Penggunaan Sepeda Berdasarkan Musim, Bulan, dan Hari dalam Seminggu
st.subheader("Pola Penggunaan Sepeda Berdasarkan Musim, Bulan, dan Hari dalam Seminggu")

# Plot Average Usage by Season, Month, and Weekday
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Average Bike Usage by Season
sns.barplot(x="season", y="cnt", data=day_df, ax=axs[0], palette="Blues")
axs[0].set_title("Average Bike Usage by Season")

# Average Bike Usage by Month
sns.barplot(x="mnth", y="cnt", data=day_df, ax=axs[1], palette="Greens")
axs[1].set_title("Average Bike Usage by Month")

# Average Bike Usage by Weekday
sns.barplot(x="weekday", y="cnt", data=day_df, ax=axs[2], palette="Reds")
axs[2].set_title("Average Bike Usage by Weekday")

st.pyplot(fig)

# Visualisasi Distribusi Penyewaan Berdasarkan Jam
st.subheader("Total Bike Rentals per Hour")
plt.figure(figsize=(12, 6))
hourly_rentals = hour_df.groupby('hr')['cnt'].sum().reset_index()
sns.barplot(x='hr', y='cnt', data=hourly_rentals, palette="viridis")
plt.title("Total Bike Rentals per Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Rentals")
st.pyplot(plt)

# Visualisasi Distribusi Pengguna Casual dan Registered Berdasarkan Kondisi Cuaca
st.subheader("Casual vs Registered Users by Weather Condition")
plt.figure(figsize=(12, 6))
sns.boxplot(x='weathersit', y='casual', data=hour_df, color="green", showfliers=False)
sns.boxplot(x='weathersit', y='registered', data=hour_df, color="purple", showfliers=False)
plt.title("Casual vs Registered Users by Weather Condition")
plt.xlabel("Weather Situation (1 = Clear, 2 = Mist, 3 = Light Snow/Rain)")
plt.ylabel("Number of Rentals")
plt.legend(['Casual Users', 'Registered Users'])
st.pyplot(plt)

# Clustering Analysis
st.subheader("Clustering Analysis of Bike Rentals")
features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(day_df[features])

# Metode Elbow untuk Clustering
st.subheader("Elbow Method for Optimal Number of Clusters")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

# Clustering dengan 3 cluster
kmeans = KMeans(n_clusters=3, random_state=42)
day_df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(12, 8))
sns.scatterplot(x='temp', y='cnt', hue='Cluster', data=day_df, palette='viridis', s=100)
plt.title('Clustering of Bike Rentals Based on Temperature and Total Rentals')
plt.xlabel('Temperature')
plt.ylabel('Total Rentals')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# Cluster Characteristics Analysis
st.subheader("Cluster Characteristics Analysis")
cluster_analysis = day_df.groupby('Cluster')[features].mean().reset_index()
st.write(cluster_analysis)

# Insight for Cluster Characteristics
st.markdown("""
**Insight:**
- Karakteristik masing-masing cluster menunjukkan bahwa hari-hari dengan suhu lebih hangat dan kelembapan rendah cenderung memiliki penyewaan sepeda yang lebih tinggi.
- Sebaliknya, hari-hari dengan suhu lebih dingin dan angin kencang memiliki penyewaan yang lebih rendah.
""")

# Kesimpulan
st.header("Kesimpulan")
st.markdown("""
Berdasarkan analisis yang telah dilakukan, terdapat beberapa insight utama yang dapat diambil:

1. **Pola Penggunaan Sepeda**: Penyewaan sepeda menunjukkan tren musiman yang jelas, dengan puncak penggunaan pada musim panas dan penurunan signifikan pada musim dingin.
   
2. **Pengaruh Cuaca**: Cuaca berperan besar dalam memengaruhi jumlah penyewaan sepeda. Hari yang lebih hangat dengan kondisi cerah mendorong lebih banyak penyewaan, sedangkan kondisi cuaca yang lebih buruk seperti hujan atau kabut mengurangi jumlah penyewaan.
   
3. **Pengguna Casual vs Registered**: Pengguna casual lebih sensitif terhadap kondisi cuaca dan lebih banyak menyewa sepeda pada siang hari. Di sisi lain, pengguna terdaftar (registered) cenderung menggunakan sepeda lebih sering pada pagi dan sore hari, terutama untuk keperluan komuter.
   
4. **Clustering**: Analisis clustering mengelompokkan hari-hari dengan penyewaan tinggi, sedang, dan rendah, berdasarkan faktor cuaca. Hari dengan suhu tinggi dan kelembapan rendah cenderung memiliki penyewaan tertinggi, sedangkan hari dengan suhu dingin dan angin kencang memiliki penyewaan yang lebih rendah.
""")
