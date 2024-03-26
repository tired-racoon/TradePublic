import os
import time

import requests

figi_list = "figi.txt"
token = YOUR_TOKEN

year = 2022
url = "https://invest-public-api.tinkoff.ru/history-data"


def download(figi, year):
    file_name = f"{figi}_{year}.zip"
    print(f"downloading {figi} for year {year}")
    response_code = requests.get(f"{url}?figi={figi}&year={year}", headers={"Authorization": f"Bearer {token}"})

    if response_code.status_code == 429:
        print("rate limit exceed. sleep 5")
        time.sleep(5)
        # download(figi, year)
        return 0

    if response_code.status_code == 401 or response_code.status_code == 500:
        print('invalid token')
        exit(1)

    if response_code.status_code == 404:
        print(f"data not found for figi={figi}, year={year}, removing empty file")

        os.remove(file_name)
    elif response_code.status_code != 200:
        print(f"unspecified error with code: {response_code.status_code}")
        exit(1)

    with open(f"{figi}_{year}.zip", "wb") as f1:
        f1.write(response_code.content)


with open(figi_list, "r") as f:
    for figi in f:
        download(figi.strip(), year)
