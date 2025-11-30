import pandas as pd
import matplotlib.pyplot as plt

#Wczytanie danych
kolumny_filmy = [
    'movie_id', 'title', 'release_date', 'video_date', 'imdb_url',
    'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
kolumny_oceny = ['user_id', 'movie_id', 'rating', 'timestamp']

# Wczytujemy pliki
filmy = pd.read_csv('u.item', sep='|', names=kolumny_filmy, encoding='latin-1')
oceny = pd.read_csv('u.data', sep='\t', names=kolumny_oceny)

# 1. Podaj liczbę filmów dla dzieci
liczba_dzieciecych = filmy[filmy['Childrens'] == 1].shape[0]
print(f"Liczba filmów dla dzieci: {liczba_dzieciecych}")

# 2. Pokaż rozkład ocen filmów z 1995
filmy_1995 = filmy[filmy['release_date'].astype(str).str.contains('1995')]

# Łączymy wybrane filmy z ocenami
oceny_1995 = pd.merge(filmy_1995, oceny, on='movie_id')

# Liczymy ile jest poszczególnych ocen i rysujemy wykres
rozklad = oceny_1995['rating'].value_counts().sort_index()
print("\nRozkład ocen filmów z 1995 roku:")
print(rozklad)

# Rysowanie prostego wykresu słupkowego
rozklad.plot(kind='bar', title='Oceny filmów z 1995')
plt.xlabel('Ocena')
plt.ylabel('Liczba głosów')
plt.show()

# 3. Podaj średnią ocen wszystkich filmów akcji oraz 3 filmy najwyżej oceniane
filmy_akcji = filmy[filmy['Action'] == 1]
oceny_akcji = pd.merge(filmy_akcji, oceny, on='movie_id')

# Średnia wszystkich ocen w tej kategorii
srednia_akcji = oceny_akcji['rating'].mean()
print(f"\nŚrednia ocena wszystkich filmów akcji: {srednia_akcji:.2f}")

# Top 3 filmy
ranking = oceny_akcji.groupby('title')['rating'].agg(['mean', 'count'])
top_3 = ranking[ranking['count'] > 50].sort_values(by='mean', ascending=False).head(3)

print("\nTop 3 najwyżej oceniane filmy akcji (min. 50 głosów):")
print(top_3)
