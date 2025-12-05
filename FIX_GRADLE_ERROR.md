# ✅ Исправлена ошибка сборки Gradle

## Проблема:
```
Build was configured to prefer settings repositories over project repositories 
but repository 'Google' was added by build file 'build.gradle'
```

## Решение:
Удалён блок `allprojects` из `build.gradle`, так как репозитории теперь управляются через `settings.gradle`.

## Что изменилось:

### До:
```gradle
buildscript { ... }

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
```

### После:
```gradle
buildscript { ... }

// allprojects удалён - репозитории в settings.gradle
```

## Следующие шаги:

```bash
cd "/Users/karimildarovic/Recept AI/FoodRecognizerApp"

# Добавить исправление
git add build.gradle

# Сделать commit
git commit -m "Fix Gradle repository configuration error"

# Отправить на GitHub
git push
```

После push, GitHub Actions автоматически пересоберёт APK. Проверьте во вкладке **Actions** через 5-10 минут.

