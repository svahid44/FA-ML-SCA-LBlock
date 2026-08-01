

from lblock import encrypt_lblock, decrypt_lblock, BLOCK_SIZE_BYTES_LBLOCK, KEY_SIZE_BYTES_LBLOCK
from present import encrypt_present, decrypt_present, BLOCK_SIZE_BYTES_PRESENT, KEY_SIZE_BYTES_PRESENT
import sys
import os
from utils import hex_to_bytes, read_file_bytes, measure_performance, write_file_bytes, display_performance_table
try:
    import psutil # Bellek ölçümü için gerekli. Yüklü değilse: pip install psutil
except ImportError:
    print("Hata: 'psutil' kütüphanesi bulunamadı.")
    print("Lütfen 'pip install psutil' komutu ile yükleyin.")
    sys.exit(1)

# GUI dosya seçimi için tkinter importları
import tkinter as tk
from tkinter import filedialog

# --- Ana Program Akışı ---

def main():
    # Tkinter ana penceresini oluştur ve gizle (sadece diyaloglar için)
    try:
        root = tk.Tk()
        root.withdraw()
        root.update() # Pencerenin gizlendiğinden emin ol
    except tk.TclError as e:
         print("Grafik arayüzü başlatılamadı (Tkinter hatası).")
         print("Program konsol modunda devam edemez.")
         print(f"Hata detayı: {e}")
         # Belki DISPLAY ortam değişkeni ayarlı değildir (Linux/WSL).
         # Veya tkinter düzgün kurulmamıştır.
         sys.exit("Tkinter başlatılamadığı için çıkılıyor.")


    print("LBlock ve PRESENT Şifreleme/Deşifreleme Performans Testi")
    print("DİKKAT: Bu kod ECB modu kullanır ve güvensizdir. Sadece eğitim/test amaçlıdır.")
    print("        Giriş verisi uzunluğu şifreleme/deşifreleme için blok boyutunun")
    print("        (LBlock/PRESENT için 8 byte) tam katı olmalıdır.")

    while True:
        print("\n" + "="*40)
        print("Menü:")
        print("1. Dosya Şifrele")
        print("2. Dosya Deşifrele")
        print("3. Çıkış")
        print("="*40)

        choice = input("Yapmak istediğiniz işlemi seçin (1/2/3): ").strip()

        if choice == '3':
            print("Çıkılıyor...")
            break

        if choice not in ['1', '2']:
            print("Geçersiz seçim. Lütfen 1, 2 veya 3 girin.")
            continue

        operation = "Şifreleme" if choice == '1' else "Deşifreleme"
        crypto_func_map = {
            '1': {'encrypt': encrypt_lblock, 'decrypt': decrypt_lblock, 'name': "LBlock",
                  'block_bytes': BLOCK_SIZE_BYTES_LBLOCK, 'key_bytes': KEY_SIZE_BYTES_LBLOCK},
            '2': {'encrypt': encrypt_present, 'decrypt': decrypt_present, 'name': "PRESENT",
                  'block_bytes': BLOCK_SIZE_BYTES_PRESENT, 'key_bytes': KEY_SIZE_BYTES_PRESENT},
        }

        print("\nAlgoritma Seçimi:")
        print("1. LBlock (80-bit anahtar, 64-bit blok)")
        print("2. PRESENT (80-bit anahtar, 64-bit blok)")
        algo_choice = input("Kullanmak istediğiniz algoritmayı seçin (1/2): ").strip()

        if algo_choice not in crypto_func_map:
            print("Geçersiz algoritma seçimi.")
            continue

        algo_details = crypto_func_map[algo_choice]
        algorithm_name = algo_details['name']
        crypto_op_func = algo_details['encrypt'] if operation == "Şifreleme" else algo_details['decrypt']
        block_size_bytes = algo_details['block_bytes']
        key_size_bytes = algo_details['key_bytes']

        print(f"\n--- {algorithm_name} {operation} ---")

        # --- Dosya Seçimi (GUI Penceresi ile) ---
        input_filepath = ""
        output_filepath = ""

        # Giriş Dosyası Seçimi
        print(f"\nLütfen {operation} yapılacak GİRİŞ dosyasını seçin...")
        # Tkinter penceresinin diğer pencerelerin üzerinde kalmasını sağla
        root.update() # Yeni diyalogdan önce güncelle
        root.deiconify() # Geçici olarak göster (bazı sistemlerde gerekebilir)
        root.lift()
        root.focus_force()
        input_filepath = filedialog.askopenfilename(
            parent=root, # Ebeveyn pencereyi belirtmek iyi olabilir
            title=f"{operation} için Giriş Dosyası Seçin",
            filetypes=[("Metin Dosyaları", "*.txt"), ("Tüm Dosyalar", "*.*")]
        )
        root.withdraw() # Tekrar gizle
        root.update()

        if not input_filepath: # Kullanıcı iptal etti veya dosya seçmedi
            print("Giriş dosyası seçilmedi. İşlem iptal edildi.")
            continue # Ana menüye dön
        print(f"Seçilen giriş dosyası: {os.path.abspath(input_filepath)}") # Tam yolu göster

        # Çıkış Dosyası Seçimi
        print(f"\nLütfen {operation} sonucunun kaydedileceği ÇIKIŞ dosyasını seçin/belirleyin...")
        root.update() # Yeni diyalogdan önce güncelle
        root.deiconify() # Geçici olarak göster
        root.lift()
        root.focus_force()
        output_filepath = filedialog.asksaveasfilename(
            parent=root,
            title=f"{operation} için Çıkış Dosyası Belirleyin",
            filetypes=[("Metin Dosyaları", "*.txt"), ("Tüm Dosyalar", "*.*")],
            defaultextension=".txt" # Kullanıcı uzantı yazmazsa bunu ekle
        )
        root.withdraw() # Tekrar gizle
        root.update()

        if not output_filepath: # Kullanıcı iptal etti
            print("Çıkış dosyası seçilmedi. İşlem iptal edildi.")
            continue # Ana menüye dön
        print(f"Seçilen çıkış dosyası: {os.path.abspath(output_filepath)}") # Tam yolu göster

        # Giriş ve Çıkış Dosyası Aynı mı Kontrolü
        # os.path.abspath kullanarak tam yolları karşılaştırmak daha güvenilir
        if os.path.abspath(input_filepath) == os.path.abspath(output_filepath):
             # Daha modern bir yaklaşım messagebox kullanmak olabilir:
             # if not messagebox.askyesno("Uyarı", f"Giriş ({os.path.basename(input_filepath)}) ve Çıkış ({os.path.basename(output_filepath)}) dosyası aynı.\nÜzerine yazılsın mı?"):
             #      print("İşlem iptal edildi.")
             #      continue
             # Şimdilik konsol ile devam:
             confirm = input(f"Uyarı: Giriş ({os.path.basename(input_filepath)}) ve Çıkış ({os.path.basename(output_filepath)}) dosyası aynı. Üzerine yazılsın mı? (e/h): ").lower().strip()
             if confirm != 'e':
                 print("İşlem iptal edildi.")
                 continue
        # --- Dosya Seçimi Sonu ---

        # Anahtar girişi (Hex formatında - Konsoldan devam)
        required_key_hex_len = key_size_bytes * 2 # 1 byte = 2 hex karakter
        key_bytes = None
        while key_bytes is None:
            key_hex = input(f"\n{algorithm_name} için {key_size_bytes} byte ({key_size_bytes * 8} bit) "
                            f"anahtarı {required_key_hex_len} karakterlik hex formatında girin "
                            f"(örn: {'0' * required_key_hex_len}): ").strip()
            key_bytes_temp = hex_to_bytes(key_hex)

            if key_bytes_temp is None:
                continue
            elif len(key_bytes_temp) != key_size_bytes:
                print(f"Hata: Girilen anahtar {len(key_bytes_temp)} byte. "
                      f"Gereken {key_size_bytes} byte ({required_key_hex_len} hex karakter).")
                continue
            else:
                key_bytes = key_bytes_temp # Anahtar geçerli

        # Dosyayı oku
        print(f"\n'{os.path.basename(input_filepath)}' dosyası okunuyor...")
        data_bytes = read_file_bytes(input_filepath)
        if data_bytes is None:
            continue # Ana menüye dön

        if len(data_bytes) == 0 and operation == "Şifreleme":
             print("Uyarı: Giriş dosyası boş. Çıkış dosyası da boş olacak.")
    

        print(f"{operation} işlemi başlatılıyor ({algorithm_name})...")
        metrics, result_bytes = measure_performance(
            operation, algorithm_name, data_bytes, key_bytes,
            crypto_op_func, block_size_bytes, key_size_bytes
        )

        if metrics and result_bytes is not None:
            print(f"{operation} tamamlandı.")
            # Sonuçları dosyaya yaz
            if write_file_bytes(output_filepath, result_bytes):
                display_performance_table([metrics])
            else:
                print("Sonuç dosyaya yazılamadığı için performans gösterilemiyor.")
        else:
            print(f"{operation} işlemi başarısız oldu veya metrik alınamadı.")


        print("\nİşlem tamamlandı. Ana menüye dönülüyor.")
        # time.sleep(1)

if __name__ == "__main__":
    main()