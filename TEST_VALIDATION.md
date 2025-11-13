# 🧪 Giriş Doğrulama Testleri (Input Validation Tests)

## ✅ Uygulanan Değişiklikler:

### 1️⃣ **Doktor Kaydı (Doctor Registration)**
- **Kullanıcı Adı (Username)**:
  - ✅ **Harf ve rakam** içermelidir
  - ❌ Sadece rakam kabul edilmez
  - ✅ Türkçe karakterlere izin verilir (ğüşıöç)
  - ✅ Minimum: 3 karakter
  - ✅ En az bir harf içermelidir

**Doğru Örnekler:**
- ✅ `ahmet123`
- ✅ `doktor_ali`
- ✅ `mehmet45`

**Yanlış Örnekler:**
- ❌ `123456` (sadece rakam)
- ❌ `12` (3 karakterden az)
- ❌ `dr@123` (özel karakterler)

---

### 2️⃣ **Hasta Bilgileri (Patient Information)**

#### 📝 **Hasta Adı (Patient Name)**
- ✅ Sadece **harfler** (rakam yok)
- ✅ Boşluklara izin verilir
- ✅ Türkçe karakterlere izin verilir
- ✅ Minimum: 2 karakter

**Doğru Örnekler:**
- ✅ `Ahmet Yılmaz`
- ✅ `Ayşe Öztürk`
- ✅ `Mehmet`

**Yanlış Örnekler:**
- ❌ `Ahmet123` (rakam içeriyor)
- ❌ `A` (sadece 1 karakter)

---

#### 🆔 **TC Kimlik No**
- ✅ **Tam 11 haneli rakam**
- ✅ Sadece rakamlar (harf yok)
- ⚠️ **Opsiyonel** (boş bırakılabilir)

**Doğru Örnekler:**
- ✅ `12345678901` (11 rakam)
- ✅ ` ` (boş - opsiyonel)

**Yanlış Örnekler:**
- ❌ `123456789` (11'den az)
- ❌ `123456789012` (11'den fazla)
- ❌ `1234567890A` (harf içeriyor)

---

#### 📞 **Telefon Numarası (Phone Number)**
- ✅ **10 haneli rakam**
- ✅ Otomatik olarak **+90** eklenir
- ✅ Herhangi bir formatta girilebilir (boşluk, tire)
- ⚠️ **Opsiyonel**

**Giriş Örnekleri:**
- `5551234567` → `+905551234567` ✅
- `555 123 45 67` → `+905551234567` ✅
- `0555 123 45 67` → `+905551234567` ✅
- `+90 555 123 45 67` → `+905551234567` ✅

**Yanlış Örnekler:**
- ❌ `555123456` (9 rakam)
- ❌ `55512345678` (11 rakam)

---

## 🔍 Çift Doğrulama (Dual Validation):

### **İstemci Tarafı (Client-Side)**
- ✅ Tarayıcıda anında doğrulama
- ✅ Net hata mesajları
- ✅ Düzeltme yapılmadan gönderim engellenir

### **Sunucu Tarafı (Server-Side)**
- ✅ Güvenlik için ikinci doğrulama
- ✅ Kullanıcıya Flash mesajları
- ✅ Hata durumunda yönlendirme

---

## 🧪 Önerilen Testler:

### 1. Doktor Kaydı Testi:
```
Username: "123456" → ❌ Hata
Username: "ahmet" → ✅ Başarılı
Username: "ahmet123" → ✅ Başarılı
```

### 2. Hasta Ekleme Testi:
```
Ad: "Ahmet Yılmaz" → ✅ Başarılı
Ad: "Ahmet123" → ❌ Hata

TC: "12345678901" → ✅ Başarılı (11 rakam)
TC: "123456789" → ❌ Hata (9 rakam)

Telefon: "5551234567" → ✅ +905551234567 olur
Telefon: "555123456" → ❌ Hata (9 rakam)
```

---

## 📋 Notlar:

1. **Tüm alanlar opsiyoneldir** (şunlar hariç):
   - ✅ Doktor adı
   - ✅ Kullanıcı adı
   - ✅ Şifre

2. **Telefon Numarası**:
   - Otomatik temizlenir
   - Otomatik +90 eklenir
   - Format: `+90XXXXXXXXXX`

3. **TC Kimlik No**:
   - **Tam 11 haneli** olmalıdır
   - Daha fazla veya daha az kabul edilmez

4. **Hasta Adı**:
   - **Sadece harfler** (Arapça, Türkçe, İngilizce)
   - **Kesinlikle rakam yok**

---

## ✅ Değiştirilen Dosyalar:

1. **app.py**:
   - Doğrulama fonksiyonları: `validate_username()`, `validate_patient_name()`, `validate_tc_kimlik()`, `validate_phone()`
   - `/register` route güncellendi
   - `/predict` route güncellendi

2. **templates/register.html**:
   - Doğrulama için `pattern` ve `title` eklendi
   - Açıklayıcı notlar eklendi

3. **templates/index.html**:
   - Hasta alanları `pattern` ve `title` ile güncellendi
   - Gönderimden önce doğrulama için JavaScript eklendi
   - Açıklayıcı mesajlar eklendi

---

## 🚀 Yayına Hazır!

Giriş doğrulama tamamlandı ✅
