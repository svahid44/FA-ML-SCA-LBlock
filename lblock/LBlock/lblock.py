from utils import bytes_to_int, int_to_bytes

# LBlock'un round fonksiyonundaki rotasyon (R <<< 8) doğrudan encrypt/decrypt içinde yapılır.
# LBlock'un key schedule'daki rotasyon (k <<< 29) doğrudan key schedule içinde yapılır.

# --- LBlock Sabitleri ve Fonksiyonları ---
SBOX_LBLOCK = [
    # S0           # S1           # S2           # S3
    [0x2, 0xa, 0xc, 0x6, 0x9, 0x0, 0x1, 0xb, 0x7, 0xd, 0x5, 0xf, 0xe, 0x4, 0x3, 0x8],
    [0xa, 0x1, 0x4, 0xc, 0x6, 0xd, 0xf, 0x7, 0x0, 0x9, 0x5, 0xe, 0x3, 0xb, 0x8, 0x2],
    [0x8, 0xc, 0x6, 0x7, 0x9, 0x3, 0xa, 0xf, 0x5, 0x1, 0xe, 0x4, 0xb, 0x0, 0xd, 0x2],
    [0x4, 0x6, 0x2, 0x9, 0x1, 0xd, 0xb, 0xe, 0x5, 0xf, 0xc, 0x7, 0xa, 0x0, 0x8, 0x3],
    # S4           # S5           # S6           # S7
    [0xb, 0x7, 0x5, 0xd, 0xf, 0x0, 0x9, 0xe, 0xa, 0x3, 0xc, 0x4, 0x8, 0x6, 0x1, 0x2],
    [0xd, 0x8, 0xb, 0x4, 0xc, 0xf, 0x3, 0x9, 0x7, 0x0, 0xa, 0xe, 0x2, 0x6, 0x1, 0x5],
    [0xf, 0x1, 0x8, 0xd, 0x6, 0xb, 0x3, 0x4, 0x9, 0x7, 0x2, 0xa, 0xc, 0x0, 0xe, 0x5],
    [0xa, 0xf, 0x6, 0x3, 0x9, 0xd, 0x1, 0x4, 0xc, 0xe, 0x0, 0x7, 0x5, 0x2, 0xb, 0x8]
]

# P Permütasyonu (32-bit giriş/çıkış) - LBlock Spesifikasyonundan Doğrulanmalı
P_PERM_LBLOCK = [
    # i=0 -> j=24, i=1 -> j=16, ...
    24, 16,  8,  0, 25, 17,  9,  1, 26, 18, 10,  2, 27, 19, 11,  3,
    28, 20, 12,  4, 29, 21, 13,  5, 30, 22, 14,  6, 31, 23, 15,  7
]

# LBlock Parametreleri
BLOCK_SIZE_BITS_LBLOCK = 64
BLOCK_SIZE_BYTES_LBLOCK = BLOCK_SIZE_BITS_LBLOCK // 8
KEY_SIZE_BITS_LBLOCK = 80
KEY_SIZE_BYTES_LBLOCK = KEY_SIZE_BITS_LBLOCK // 8
NUM_ROUNDS_LBLOCK = 32

# --- Anahtar Planı (LBlock) ---
def key_schedule_lblock(master_key_int):
    """80-bit anahtardan 32 adet 32-bit round anahtarı üretir."""
    if not (0 <= master_key_int < (1 << KEY_SIZE_BITS_LBLOCK)):
        raise ValueError("LBlock master key integer değeri 80-bit aralığında olmalı.")

    round_keys = [0] * NUM_ROUNDS_LBLOCK
    k = master_key_int # 80-bit anahtar

    mask_80 = (1 << 80) - 1 # 80 bitlik maske
    mask_32 = (1 << 32) - 1 # 32 bitlik maske

    for i in range(1, NUM_ROUNDS_LBLOCK + 1):
        # 4. Round anahtarını al (Anahtarın sol 32 biti) - ÖNCE alınır
        round_keys[i-1] = (k >> (80 - 32)) & mask_32

        # Anahtar güncelleme adımları (bir sonraki round için)
        # 1. Döndürme: k = ROL(k, 29, 80)
        k = ((k << 29) | (k >> (80 - 29))) & mask_80

        # 2. S-Box Uygulaması (Anahtarın üst 8 bitine: bit 79..76 ve 75..72)
        # Üst 4 bit (bit 79..76) -> SBOX[7]
        # Sonraki 4 bit (bit 75..72) -> SBOX[6]
        s_in1 = (k >> 76) & 0xF # Bit 79..76
        s_in2 = (k >> 72) & 0xF # Bit 75..72
        s_out1 = SBOX_LBLOCK[7][s_in1]
        s_out2 = SBOX_LBLOCK[6][s_in2]
        # Eski üst 8 biti temizle ve yeni S-Box çıkışlarını yerleştir
        k = (k & ((1 << 72) - 1)) | (s_out1 << 76) | (s_out2 << 72)

        # 3. Round Counter XOR (bit 19..15 ile)
        # LBlock spec: i XORed with k[63..59] which corresponds to bits 15-19 from the right end
        round_counter = i & 0x1F # 5 bit (1 to 32)
        # Bit 15-19'u etkilemek için 15 bit sola kaydır
        xor_val = round_counter << (64 - 5) # LBlock spec'e göre üst bitlere XORlanır (k[63]..k[59]) bu da sağdan 15-19 arası demek
        # Düzeltme: Makaledeki K[19..15] notasyonu genellikle sağdan sayar.
        # Bu durumda 15 bit sola kaydırmak doğrudur.
        k ^= (round_counter << 15)
        k &= mask_80 # Taşmaları önle

    return round_keys

# --- Round Fonksiyonu F (LBlock) ---
def function_F_lblock(r_half, round_key):
    """LBlock F fonksiyonu (32-bit giriş/anahtar, 32-bit çıkış)."""
    temp = r_half ^ round_key # 1. Anahtar ile XOR
    s_output = 0
    # 2. S-Box Katmanı
    # 8 adet 4-bitlik bloğa 8 farklı S-Box uygulanır.
    # S7 -> en anlamlı (sol) 4-bit (bit 31..28)
    # S0 -> en anlamsız (sağ) 4-bit (bit 3..0)
    for i in range(8):
        # Sağdan i. 4-bitlik nibble'ı al (i=0 -> bit 3..0, i=7 -> bit 31..28)
        start_bit = i * 4
        nibble = (temp >> start_bit) & 0xF
        # İlgili S-Box'ı kullan (S0'dan S7'ye doğru)
        s_out = SBOX_LBLOCK[i][nibble]
        s_output |= (s_out << start_bit)

    # 3. P-Permütasyon Katmanı
    p_output = 0
    for i in range(32):
        if (s_output >> i) & 1: # Eğer s_output'un i. biti 1 ise
            # Çıkışın P_PERM_LBLOCK[i]. bitini 1 yap
            p_output |= (1 << P_PERM_LBLOCK[i])
    return p_output

# --- Şifreleme (LBlock) ---
def encrypt_block_lblock(plaintext_int, round_keys):
    """Tek bir 64-bit bloğu LBlock ile şifreler."""
    mask_32 = (1 << 32) - 1
    # Başlangıç L0, R0
    L = (plaintext_int >> 32) & mask_32
    R = plaintext_int & mask_32

    # 32 Round
    for i in range(NUM_ROUNDS_LBLOCK):
        F_result = function_F_lblock(R, round_keys[i])
        # LBlock round yapısı: L_i+1 = R_i; R_i+1 = (L_i ^ F(R_i, K_i)) <<< 8
        new_R_unrotated = L ^ F_result
        # 8 bit sola dairesel kaydırma (ROL)
        new_R = ((new_R_unrotated << 8) | (new_R_unrotated >> (32 - 8))) & mask_32
        new_L = R

        L, R = new_L, new_R

    # Sonuç (L32 ve R32'yi birleştir, L solda, R sağda)
    ciphertext_int = (L << 32) | R
    return ciphertext_int

# --- Ana Şifreleme Fonksiyonu (Byte Girdi/Çıktı - LBlock) ---
def encrypt_lblock(plaintext_bytes, key_bytes):
    """Byte dizisi girdiyi LBlock ile şifreler (ECB modu).
       Girdi uzunluğu blok boyutunun katı değilse, sonuna boşluk (0x20) byte'ları eklenir.
       DİKKAT: ECB modu güvensizdir. Boşluk padding'i standart değildir.
    """
    if len(key_bytes) != KEY_SIZE_BYTES_LBLOCK:
        raise ValueError(f"LBlock Şifreleme Hatası: Anahtar uzunluğu ({len(key_bytes)} bytes) "
                         f"{KEY_SIZE_BYTES_LBLOCK} bytes olmalı.")

    master_key_int = bytes_to_int(key_bytes)
    try:
        round_keys = key_schedule_lblock(master_key_int)
    except ValueError as e:
        raise ValueError(f"LBlock Anahtar Planı Hatası: {e}") from e

    # --- Padding Başlangıcı ---
    original_length = len(plaintext_bytes)
    remainder = original_length % BLOCK_SIZE_BYTES_LBLOCK
    padded_plaintext_bytes = plaintext_bytes # Başlangıçta aynı

    if remainder != 0:
        bytes_to_add = BLOCK_SIZE_BYTES_LBLOCK - remainder
        # Boşluk karakteri (ASCII 32, hex 0x20) byte'ları ile doldur
        padding = b' ' * bytes_to_add
        padded_plaintext_bytes += padding
        print(f"Bilgi (LBlock): Girdi {original_length} byte idi. {bytes_to_add} boşluk byte'ı ile dolduruldu. Yeni boyut: {len(padded_plaintext_bytes)}")
    # --- Padding Sonu ---

    if len(padded_plaintext_bytes) == 0:
         # Padding sonrası hala boşsa (yani orijinal girdi de boşsa)
         print("Uyarı (LBlock): Şifrelenecek veri boş.")
         return b''

    # Artık padded_plaintext_bytes uzunluğu kesinlikle 8'in katı olmalı
    ciphertext_bytes = b''
    for i in range(0, len(padded_plaintext_bytes), BLOCK_SIZE_BYTES_LBLOCK):
        block = padded_plaintext_bytes[i : i + BLOCK_SIZE_BYTES_LBLOCK]
        block_int = bytes_to_int(block)
        encrypted_block_int = encrypt_block_lblock(block_int, round_keys)
        encrypted_block_bytes = int_to_bytes(encrypted_block_int, BLOCK_SIZE_BYTES_LBLOCK)
        if encrypted_block_bytes is None:
             raise ValueError("LBlock Şifreleme Hatası: Şifrelenmiş blok integer'dan byte'a dönüştürülemedi.")
        ciphertext_bytes += encrypted_block_bytes

    return ciphertext_bytes

# --- Deşifreleme (LBlock) ---
def decrypt_block_lblock(ciphertext_int, round_keys):
    """Tek bir 64-bit bloğu LBlock ile deşifreler."""
    mask_32 = (1 << 32) - 1
    # Başlangıç L32, R32 (Şifrelenmiş blok)
    L = (ciphertext_int >> 32) & mask_32
    R = ciphertext_int & mask_32

    # 32 Round (Ters sırada, round 31'den 0'a)
    for i in range(NUM_ROUNDS_LBLOCK - 1, -1, -1):
        # Şifreleme adımlarını tersine çevir:
        # L_i+1 = R_i; R_i+1 = (L_i ^ F(R_i, K_i)) <<< 8
        # Başlangıç: L = L_(i+1), R = R_(i+1)
        # 1. Rotasyonu geri al: R' = R_(i+1) >>> 8
        # 2. F fonksiyonunu hesapla: F_val = F(L_(i+1), K_i) = F(R_i, K_i)
        # 3. L_i'yi bul: R' = L_i ^ F_val => L_i = R' ^ F_val
        # 4. R_i'yi bul: R_i = L_(i+1)

        # Adım 1: Rotasyonu geri al (R >>> 8)
        rotated_R = ((R >> 8) | (R << (32 - 8))) & mask_32

        # Adım 2: F fonksiyonunu hesapla (Girdi olarak mevcut L kullanılır, bu L_(i+1) == R_i)
        F_result = function_F_lblock(L, round_keys[i])

        # Adım 3: Önceki L'yi (L_i) hesapla
        prev_L = rotated_R ^ F_result

        # Adım 4: Önceki R'yi (R_i) hesapla
        prev_R = L

        # Bir sonraki deşifreleme adımı için state'i güncelle
        L, R = prev_L, prev_R

    # Sonuç (L0 ve R0'ı birleştir)
    plaintext_int = (L << 32) | R
    return plaintext_int

def decrypt_lblock(ciphertext_bytes, key_bytes):
    """Byte dizisi girdiyi LBlock ile deşifreler (ECB modu)."""
    if len(ciphertext_bytes) == 0:
        print("Uyarı: Deşifrelenecek veri boş.")
        return b''
    if len(ciphertext_bytes) % BLOCK_SIZE_BYTES_LBLOCK != 0:
        raise ValueError(f"LBlock Deşifreleme Hatası: Şifrelenmiş metin uzunluğu ({len(ciphertext_bytes)} bytes) "
                         f"blok boyutunun ({BLOCK_SIZE_BYTES_LBLOCK} bytes) katı olmalı (ECB modu).")
    if len(key_bytes) != KEY_SIZE_BYTES_LBLOCK:
        raise ValueError(f"LBlock Deşifreleme Hatası: Anahtar uzunluğu ({len(key_bytes)} bytes) "
                         f"{KEY_SIZE_BYTES_LBLOCK} bytes olmalı.")

    master_key_int = bytes_to_int(key_bytes)
    try:
        round_keys = key_schedule_lblock(master_key_int)
    except ValueError as e:
        raise ValueError(f"LBlock Anahtar Planı Hatası: {e}") from e

    plaintext_bytes = b''
    for i in range(0, len(ciphertext_bytes), BLOCK_SIZE_BYTES_LBLOCK):
        block = ciphertext_bytes[i : i + BLOCK_SIZE_BYTES_LBLOCK]
        block_int = bytes_to_int(block)
        decrypted_block_int = decrypt_block_lblock(block_int, round_keys)
        decrypted_block_bytes = int_to_bytes(decrypted_block_int, BLOCK_SIZE_BYTES_LBLOCK)
        if decrypted_block_bytes is None:
             raise ValueError("LBlock Deşifreleme Hatası: Deşifrelenmiş blok integer'dan byte'a dönüştürülemedi.")
        plaintext_bytes += decrypted_block_bytes

    return plaintext_bytes