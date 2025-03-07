from bs4 import BeautifulSoup
import requests
import datetime
import os
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


URL_BASE = 'https://www.soaringspot.com'
PYSOAR_OUT = '/Users/jkretzs/Desktop/pysoar/bin'
PYSOAR_BIN = '/Users/jkretzs/personal/scripts/PySoar/PySoar/main_pysoar.py'


def main():

    # Language of soaringspot
    lang = 'de'

    # Name of competition and URLs

    comp = 'omv-2022'
    url_comp = f'{URL_BASE}/{lang}/{comp}'
    sel_class = 'doppelsitzer'

    # URLs of all competition days for all classes
    compday_urls(url_comp, sel_class)


def compday_urls(url_comp, sel_class_in):
    url_comp_results = f'{url_comp}/results'

    import pickle

    # Set up session
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    req = session.get(url_comp_results)
    # with open('tmp.pkl', 'wb') as f:
    #     pickle.dump(req, f)
    # exit()
    #
    # with open('tmp.pkl', 'rb') as f:
    #     req = pickle.load(f)

    soup = BeautifulSoup(req.text, "html.parser")
    result_class_all = soup.find_all('table', class_='result-overview')

    urls_dict = {}
    for comp_class in result_class_all:
        if sel_class_in not in str(comp_class.contents[1]):
            continue
        results_class = comp_class.find_all('tr')
        days_class = []
        for element in results_class:
            urls = element.find_all('a')
            if len(urls) == 3:
                for url in urls:
                    if 'daily' in url.attrs['href']:
                        days_class.append(url.attrs['href'])

        for day_url in days_class:
            day_url_content = day_url.split('/')
            comp = day_url_content[2]
            comp_class = day_url_content[4]
            day_datetime = datetime.datetime.strptime(day_url_content[5][-10::], '%Y-%m-%d')
            day = datetime.datetime.strftime(day_datetime, '%d-%m-%Y')
            opath = f'{PYSOAR_OUT}/{comp}/{comp_class}/{day}/Analysis_PySoar.xls'

            if not os.path.isfile(opath):
                print(comp, comp_class, day)
                day_url = URL_BASE+day_url
                os.system(f'python {PYSOAR_BIN} {day_url}')


if __name__ == "__main__":
    main()


exit()
