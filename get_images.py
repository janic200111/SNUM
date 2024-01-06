from selenium import webdriver
import time
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_google_images(query, num_images):
    # Ścieżka do sterownika przeglądarki, na przykład chromedriver
    driver_path = "C:\chromedriver\chromedriver.exe"
     # Ustawienia opcji dla przeglądarki Chrome
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")  # Opcjonalnie: uruchom w trybie bez głowy
    chrome_options.add_argument(f"executable_path={driver_path}")

    # Utwórz instancję przeglądarki Chrome z użyciem opcji
    browser = webdriver.Chrome(options=chrome_options)

    # Otwórz stronę Google Images
    browser.get("https://www.google.com/imghp")

    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.NAME, "q")))

    search_box = browser.find_element("id", "some_id")
    search_box.send_keys(query)
    search_box.submit()

    # Poczekaj na załadowanie wyników
    time.sleep(2)

    # Przewiń stronę, aby załadować więcej obrazów
    for _ in range(num_images // 50):
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # Pobierz adresy URL obrazów
    img_urls = [img.get_attribute("src") for img in browser.find_elements_by_css_selector("img.rg_i")]

    # Pobierz obrazy
    for i, img_url in enumerate(img_urls[:num_images]):
        os.system(f"wget -O {query}_{i+1}.jpg {img_url}")

    browser.quit()

if __name__ == "__main__":
    download_google_images("zdjecia", 10)
