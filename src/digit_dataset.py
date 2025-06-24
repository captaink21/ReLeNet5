import numpy as np
import torch
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageChops, ImageFilter
from sklearn.model_selection import train_test_split


def augmentations(image):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    dx, dy = random.randint(-2, 2), random.randint(-1, 1)
    image = ImageChops.offset(image, dx, dy)

    image = image.rotate(random.uniform(-10, 10),
                         resample=Image.BICUBIC, fillcolor=255)

    arr = np.array(image)
    noise_level = random.choice([0.01, 0.02, 0.03])
    noise = np.random.randint(0, 40, arr.shape) * \
        (np.random.rand(*arr.shape) < noise_level)
    noisy = np.clip(arr + noise, 0, 255).astype('uint8')
    image = Image.fromarray(noisy)

    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.5, 1.2)))

    return image


def generate_digit_image(digit, system="western", augment=True):

    if system == "western":
        char = str(digit)
    elif system == "eastern_arabic":
        eastern_arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        char = eastern_arabic_digits[digit]
    elif system == "roman":
        roman_digits = ["N", "I", "II", "III",
                        "IV", "V", "VI", "VII", "VIII", "IX"]
        char = roman_digits[digit]
    else:
        char = str(digit)

    img = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(img)

    try:
        font_size = random.randint(18, 22)
        font = ImageFont.truetype(random.choice(
            ["arial.ttf", "times.ttf", "verdana.ttf"]), font_size)
    except IOError:
        font = ImageFont.load_default()
        font_size = 20

    x = random.randint(8, 12)
    y = random.randint(2, 5)
    draw.text((x, y), char, font=font, fill=0)

    if augment:
        img = augmentations(img)

    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)

    return img_array / 255.0, char, img

    """
    samples_per_digit: Количество образцов на каждую цифру;
    test_size: Доля данных для тестовой выборки;
    mislabel_hard_fraction: Доля цифр, которые имеют подмененное значение.
    """


def generate_separate_datasets(output_dir="digit_dataset", samples_per_digit=500, test_size=0.2, mislabel_hard_fraction=0.5, save_images=False):

    os.makedirs(output_dir, exist_ok=True)

    systems = ["western", "eastern_arabic", "roman"]
    hard_fraction = 0.25
    hard_samples = int(samples_per_digit * hard_fraction)
    normal_samples = samples_per_digit - hard_samples

    if save_images:
        images_output_dir = os.path.join(output_dir, "images")
        os.makedirs(images_output_dir, exist_ok=True)
        print(f"Изображения будут сохраняться в: {images_output_dir}")

    for system in systems:
        X_all_data = []  # (img_array, img_obj)
        y_all_labels = []

        if save_images:
            system_images_dir = os.path.join(images_output_dir, system)
            os.makedirs(system_images_dir, exist_ok=True)
            train_images_dir = os.path.join(system_images_dir, "train")
            test_images_dir = os.path.join(system_images_dir, "test")
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(test_images_dir, exist_ok=True)

            img_counter_train = 0
            img_counter_test = 0

        print(
            f"--- Генерация датасета для {system.replace('_', ' ').title()} чисел ---")
        for digit in range(10):
            for _ in range(normal_samples):
                img_array, _, img_obj = generate_digit_image(
                    digit, system=system)
                X_all_data.append((img_array, img_obj))
                y_all_labels.append(digit)

            for _ in range(hard_samples):
                should_mislabel = random.random() < mislabel_hard_fraction

                if should_mislabel:
                    mislabel_digit = random.choice(
                        [d for d in range(10) if d != digit])
                    img_array, _, img_obj_base = generate_digit_image(
                        mislabel_digit, system=system)
                    true_label = digit
                else:
                    img_array, _, img_obj_base = generate_digit_image(
                        digit, system=system)
                    true_label = digit

                img_obj_hard_augmented = img_obj_base

                if random.random() > 0.5:
                    img_obj_hard_augmented = img_obj_hard_augmented.filter(
                        ImageFilter.GaussianBlur(radius=random.uniform(1.5, 2.5)))
                else:
                    arr = np.array(img_obj_hard_augmented)
                    noise = np.random.randint(
                        0, 80, arr.shape) * (np.random.rand(*arr.shape) < 0.1)
                    img_obj_hard_augmented = Image.fromarray(
                        np.clip(arr + noise, 0, 255).astype("uint8"))

                img_obj_hard_augmented = img_obj_hard_augmented.resize(
                    (28, 28), Image.LANCZOS)
                img_array_final = np.array(
                    img_obj_hard_augmented, dtype=np.uint8) / 255.0

                X_all_data.append((img_array_final, img_obj_hard_augmented))
                y_all_labels.append(true_label)

            print(
                f"  [{system}] Цифра {digit}: сгенерировано {samples_per_digit} примеров.")

        X_arrays = np.array([item[0] for item in X_all_data])
        X_images = [item[1] for item in X_all_data]

        X_train_np, X_test_np, y_train_np, y_test_np, X_train_images, X_test_images = train_test_split(
            X_arrays, np.array(y_all_labels), X_images, test_size=test_size, random_state=42, stratify=y_all_labels
        )

        if save_images:
            for idx, label in enumerate(y_train_np):
                img_to_save = X_train_images[idx]
                img_filename = os.path.join(
                    train_images_dir, f"{system}_{label}_{img_counter_train:05d}.png")
                img_to_save.save(img_filename)
                img_counter_train += 1

            for idx, label in enumerate(y_test_np):
                img_to_save = X_test_images[idx]
                img_filename = os.path.join(
                    test_images_dir, f"{system}_{label}_{img_counter_test:05d}.png")
                img_to_save.save(img_filename)
                img_counter_test += 1
            print(
                f"  [{system.replace('_', ' ').title()}] Сохранено {img_counter_train} изображений в train и {img_counter_test} в test.")

        X_train_tensor = torch.tensor(
            X_train_np[:, None, :, :], dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
        X_test_tensor = torch.tensor(
            X_test_np[:, None, :, :], dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)

        torch.save(X_train_tensor, os.path.join(
            output_dir, f"{system}_X_train.pt"))
        torch.save(y_train_tensor, os.path.join(
            output_dir, f"{system}_y_train.pt"))
        torch.save(X_test_tensor, os.path.join(
            output_dir, f"{system}_X_test.pt"))
        torch.save(y_test_tensor, os.path.join(
            output_dir, f"{system}_y_test.pt"))

        print(f"✅ [{system.replace('_', ' ').title()}] сохранено: {len(X_train_tensor)} train samples, {len(X_test_tensor)} test samples.")
        print("-" * 100)


if __name__ == "__main__":
    dataset_output_directory = "digit_dataset"

    generate_separate_datasets(
        output_dir=dataset_output_directory,
        samples_per_digit=500,
        test_size=0.2,
        mislabel_hard_fraction=0.4,
        save_images=True  # Формирование картинок
    )

    print(
        f"\nВсе датасеты сгенерированы и сохранены в папке: {dataset_output_directory}")
