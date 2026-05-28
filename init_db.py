import sqlite3

def init_db():
    conn = sqlite3.connect("cv_analytics.db")
    cursor = conn.cursor()

    # Удаляем старую таблицу, чтобы создать новую с правильной структурой
    cursor.execute("DROP TABLE IF EXISTS people_tracks")

    # Создаем новую таблицу
    # track_id не уникален глобально (он может повториться через долгое время), 
    # поэтому добавим автоинкрементный id как первичный ключ
    cursor.execute("""
                   CREATE TABLE people_tracks
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       track_id INTEGER NOT NULL,
                       label TEXT NOT NULL,
                       timestamp DATETIME DEFAULT (datetime('now', 'localtime'))
                   )
                   """)

    conn.commit()
    conn.close()
    print("База данных успешно пересоздана (структура: id, track_id, label, timestamp)!")

if __name__ == "__main__":
    init_db()
