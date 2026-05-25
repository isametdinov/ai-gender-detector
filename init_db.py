import sqlite3


def init_db():
    # Создаст файл базы данных в папке проекта
    conn = sqlite3.connect("cv_analytics.db")
    cursor = conn.cursor()

    # Создаем таблицу для детекций
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS people_tracks
                   (
                       track_id
                       INTEGER
                       PRIMARY
                       KEY,
                       gender_age
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       DATETIME
                       DEFAULT (
                       datetime
                   (
                       'now',
                       'localtime'
                   ))
                       )
                   """)

    conn.commit()
    conn.close()
    print("База данных успешно создана!")


if __name__ == "__main__":
    init_db()