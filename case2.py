# Instalacja pakietów
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# Wczytanie danych
kolumny_filmy = [
    'movie_id', 'title', 'release_date', 'video_date', 'imdb_url',
    'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
kolumny_oceny = ['user_id', 'movie_id', 'rating', 'timestamp']

filmy = pd.read_csv('u.item', sep='|', names=kolumny_filmy, encoding='latin-1')
oceny = pd.read_csv('u.data', sep='\t', names=kolumny_oceny)

# Podaj liczbę filmów dla dzieci
liczba_dzieciecych = filmy[filmy['Childrens'] == 1].shape[0]
print(f"Liczba filmów dla dzieci: {liczba_dzieciecych}")

# Pokaż rozkład ocen filmów z 1995
filmy_1995 = filmy[filmy['release_date'].astype(str).str.contains('1995')]
oceny_1995 = pd.merge(filmy_1995, oceny, on='movie_id')
rozklad = oceny_1995['rating'].value_counts().sort_index()
print("\nRozkład ocen filmów z 1995 roku:")
print(rozklad)

# Podaj średnią ocen wszystkich filmów akcji oraz 3 filmy najwyżej oceniane
filmy_akcji = filmy[filmy['Action'] == 1]
oceny_akcji = pd.merge(filmy_akcji, oceny, on='movie_id')
srednia_akcji = oceny_akcji['rating'].mean()
print(f"\nŚrednia ocena wszystkich filmów akcji: {srednia_akcji:.2f}")

ranking = oceny_akcji.groupby('title')['rating'].agg(['mean', 'count'])
top_3 = ranking[ranking['count'] > 50].sort_values(by='mean', ascending=False).head(3)
print("\nTop 3 najwyżej oceniane filmy akcji (min. 50 głosów):")
print(top_3)

# Zbuduj system na podstawie algorytmu SVD oraz kNNwithMeans
print("SYSTEM REKOMENDACYJNY")


macierz_ocen = oceny.pivot(index='user_id', columns='movie_id', values='rating')
macierz_ocen_filled = macierz_ocen.fillna(0)

# SVD
print("\n[SVD] Generowanie modelu...")
svd = TruncatedSVD(n_components=20, random_state=42)
macierz_zredukowana = svd.fit_transform(macierz_ocen_filled)
macierz_predykcji_svd = svd.inverse_transform(macierz_zredukowana)
df_predykcje_svd = pd.DataFrame(macierz_predykcji_svd, index=macierz_ocen.index, columns=macierz_ocen.columns)


def rekomenduj_svd(user_id, liczba_rekomendacji=5):
    user_pred = df_predykcje_svd.loc[user_id].sort_values(ascending=False)
    juz_widzial = oceny[oceny['user_id'] == user_id]['movie_id'].tolist()
    rekomendacje = user_pred[~user_pred.index.isin(juz_widzial)].head(liczba_rekomendacji)
    wynik = filmy[filmy['movie_id'].isin(rekomendacje.index)][['movie_id', 'title']]
    wynik['przewidywana_ocena'] = wynik['movie_id'].map(rekomendacje)
    return wynik.sort_values(by='przewidywana_ocena', ascending=False)


# GridSearch
print("\n[Grid Search] Szukanie najlepszego 'k'...")

train_data, test_data = train_test_split(oceny, test_size=0.1, random_state=42)
train_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
test_data = test_data[test_data['user_id'].isin(train_matrix.index)]
test_data = test_data[test_data['movie_id'].isin(train_matrix.columns)]
test_sample = test_data.sample(n=min(1000, len(test_data)), random_state=42)

najlepsze_k = 5
najmniejsze_rmse = float('inf')

for k in range(3, 11):
    model_gs = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k)
    model_gs.fit(train_matrix)
    bledy = []

    for _, row in test_sample.iterrows():
        u_id, m_id, rating_real = row['user_id'], row['movie_id'], row['rating']
        user_vec = train_matrix.loc[u_id].values.reshape(1, -1)
        distances, indices = model_gs.kneighbors(user_vec, n_neighbors=k)
        sasiedzi_idx = indices.flatten()
        sasiedzi_oceny = train_matrix.iloc[sasiedzi_idx][m_id]
        oceny_niezerowe = sasiedzi_oceny[sasiedzi_oceny > 0]

        if len(oceny_niezerowe) > 0:
            predykcja = oceny_niezerowe.mean()
        else:
            predykcja = train_matrix.loc[u_id].mean()

        bledy.append((predykcja - rating_real) ** 2)

    rmse = np.sqrt(np.mean(bledy))
    print(f"   -> k={k}: RMSE = {rmse:.4f}")
    if rmse < najmniejsze_rmse:
        najmniejsze_rmse = rmse
        najlepsze_k = k

print(f"Najlepsze k: {najlepsze_k}")

# kNNwithMeans
knn_final = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=najlepsze_k)
knn_final.fit(macierz_ocen_filled)


def rekomenduj_knn(user_id, liczba_rekomendacji=5):
    user_vector = macierz_ocen_filled.loc[user_id].values.reshape(1, -1)
    distances, indices = knn_final.kneighbors(user_vector, n_neighbors=najlepsze_k + 1)
    neighbor_indices = indices.flatten()[1:]
    similar_users = macierz_ocen_filled.iloc[neighbor_indices]
    mean_ratings = similar_users.replace(0, np.nan).mean(axis=0)
    juz_widzial = oceny[oceny['user_id'] == user_id]['movie_id'].tolist()
    mean_ratings = mean_ratings.drop(juz_widzial, errors='ignore')
    top_movie_ids = mean_ratings.sort_values(ascending=False).head(liczba_rekomendacji).index
    wynik = filmy[filmy['movie_id'].isin(top_movie_ids)][['movie_id', 'title']]
    wynik['srednia_ocena_sasiadow'] = wynik['movie_id'].map(mean_ratings)
    return wynik.sort_values(by='srednia_ocena_sasiadow', ascending=False)


# Walidacja krzyżowa
print("Walidacja Krzyżowa (5 Folds)")


kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_svd_folds = []
rmse_knn_folds = []

fold_nr = 1
for train_index, test_index in kf.split(oceny):
    train_df = oceny.iloc[train_index]
    test_df = oceny.iloc[test_index]
    train_matrix_fold = train_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    test_df_clean = test_df[test_df['user_id'].isin(train_matrix_fold.index)]
    test_df_clean = test_df_clean[test_df_clean['movie_id'].isin(train_matrix_fold.columns)]
    test_sample_fold = test_df_clean.sample(n=min(500, len(test_df_clean)), random_state=42)

    svd_cv = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = svd_cv.fit_transform(train_matrix_fold)
    matrix_reconstructed = svd_cv.inverse_transform(matrix_reduced)
    df_reconstructed = pd.DataFrame(matrix_reconstructed, index=train_matrix_fold.index,
                                    columns=train_matrix_fold.columns)
    errors_svd = []
    for _, row in test_sample_fold.iterrows():
        pred = df_reconstructed.loc[row['user_id'], row['movie_id']]
        errors_svd.append((pred - row['rating']) ** 2)
    rmse_svd_folds.append(np.sqrt(np.mean(errors_svd)))

    knn_cv = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=najlepsze_k)
    knn_cv.fit(train_matrix_fold)
    errors_knn = []
    for _, row in test_sample_fold.iterrows():
        u_vec = train_matrix_fold.loc[row['user_id']].values.reshape(1, -1)
        dists, idxs = knn_cv.kneighbors(u_vec, n_neighbors=najlepsze_k)
        neighbors_idx = idxs.flatten()
        neighbors_ratings = train_matrix_fold.iloc[neighbors_idx][row['movie_id']]
        valid_ratings = neighbors_ratings[neighbors_ratings > 0]
        if len(valid_ratings) > 0:
            pred_knn = valid_ratings.mean()
        else:
            pred_knn = train_matrix_fold.loc[row['user_id']].mean()
        errors_knn.append((pred_knn - row['rating']) ** 2)
    rmse_knn_folds.append(np.sqrt(np.mean(errors_knn)))

    print(f"Fold {fold_nr}/5 zakończony.")
    fold_nr += 1

print(f"\nŚredni RMSE dla SVD (5-fold CV): {np.mean(rmse_svd_folds):.4f}")
print(f"Średni RMSE dla kNN (k={najlepsze_k}, 5-fold CV): {np.mean(rmse_knn_folds):.4f}")

# Podaj rekomendacje po obejrzeniu filmu: NeverEnding Story III oraz Pi
print("REKOMENDACJE DLA KONKRETNYCH FILMÓW (Item-Based)")


macierz_filmow = macierz_ocen_filled.T
knn_item = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
knn_item.fit(macierz_filmow)


def rekomenduj_po_tytule(czesc_tytulu):
    znalezione = filmy[filmy['title'].str.contains(czesc_tytulu, case=False, regex=False)]

    if znalezione.empty:
        print(f"\n[BŁĄD] Nie znaleziono filmu zawierającego frazę: '{czesc_tytulu}'")
        return

    movie_id = znalezione.iloc[0]['movie_id']
    tytul_filmu = znalezione.iloc[0]['title']

    print(f"\nPonieważ obejrzałeś '{tytul_filmu}', może Ci się spodobać:")

    movie_vec = macierz_filmow.loc[movie_id].values.reshape(1, -1)
    distances, indices = knn_item.kneighbors(movie_vec, n_neighbors=6)
    neighbor_ids = macierz_filmow.iloc[indices.flatten()[1:]].index
    wyniki = filmy[filmy['movie_id'].isin(neighbor_ids)][['title']]
    print(wyniki.to_string(index=False, header=False))


rekomenduj_po_tytule("NeverEnding Story III")
rekomenduj_po_tytule("Pi (1998)")

min_id = macierz_ocen.index.min()
max_id = macierz_ocen.index.max()


print(f" Dostępne ID w bazie: od {min_id} do {max_id}")

while True:
    try:
        user_input = input(f"\nPodaj ID użytkownika (lub 'exit'): ")

        if user_input.lower() == 'exit':
            print("Zamykanie programu...")
            break

        user_testowy = int(user_input)

        if user_testowy in macierz_ocen.index:
            print(f"\n [SVD] Rekomendacje dla Usera {user_testowy} ")
            print(rekomenduj_svd(user_testowy))

            print(f"\n [kNN] Rekomendacje dla Usera {user_testowy} (Najlepsze k={najlepsze_k}) ")
            print(rekomenduj_knn(user_testowy))

            kontynuacja = input("\nInny użytkownik? (t/n): ")
            if kontynuacja.lower() != 't':
                break
        else:
            print(f"Błąd: Podaj ID z zakresu {min_id}-{max_id}.")

    except ValueError:
        print("Błąd: Wpisz liczbę całkowitą.")