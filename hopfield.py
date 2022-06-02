"""
Example of use Hopfield Recurrent network
=========================================

Task: Recognition of Simpsons

"""
import io
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from PIL import Image
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi

TEST_PATH = './data/test/*.png'
TRAIN_PATH = './data/train/*.png'
REF_PATH = './data/reference/*.png'


def show_images(images):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(images[0])
    ax.set_title('Test')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(images[1])

    ax.set_title('Result')

    plt.show()


# Utility function, which changes xi from the hopfield network library to a PIL image
def xi_to_PIL(xi, N):
    N_sqrt = int(np.sqrt(N))
    image_arr = np.uint8((xi.reshape(N_sqrt, N_sqrt) + 1) * 90)
    plt.matshow(image_arr, cmap="Blues")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    # img_buf.close()
    return im


def noise_image(file, percent):
    img = Image.open(file).convert('L')
    img = np.array(img)
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            if randint(0, 100) < percent:
                img[i][j] = 255 - img[i][j]
    img = Image.fromarray(img)

    file = file.split(".png")[0]
    file = file.split("\\")[1]
    filename = ".\\data\\test\\" + file + str(percent) + ".png"

    img.save(filename)


def get_train_images_as_reference():
    for file in glob.glob(TRAIN_PATH):
        img_test = images2xi([file], N)
        network.set_initial_neurons_state(np.copy(img_test[:, 0]))
        network.update_neurons(iterations, 'sync')

        test_pil = xi_to_PIL(img_test[:, 0], N)
        res_pil = xi_to_PIL(network.S, N)

        file = file.split(".png")[0]
        file = file.split("\\")[1]
        filename = ".\\data\\reference\\" + file + ".png"
        test_pil.save(filename)


def predict(res_pil):
    mses = {}
    res_pil.save("result.png")
    res_pil = Image.open("result.png").convert('L')
    for file in glob.glob(REF_PATH):
        reference_img = Image.open(file).convert('L')
        mses[file] = mse_img(res_pil, reference_img)

    lowest_mse = float('inf')
    lowest_name = None
    for filename, mse in mses.items():
        if mse < lowest_mse:
            lowest_mse = mse
            lowest_name = filename

    lowest_name = lowest_name.split("\\")[1]
    lowest_name = lowest_name.split(".png")[0]
    lowest_name = lowest_name.split("_")[0]
    print(f"Predykcja -> {lowest_name.capitalize()} z mse = {lowest_mse}")
    return lowest_name


def mse_img(img_true, img_pred):
    img_true, img_pred = np.array(img_true), np.array(img_pred)
    w, h = img_pred.shape[:2]
    values0 = 0
    for x in range(w):
        for y in range(h):
            if img_pred[x][y] == 0:
                values0 += 1

    if values0 > w*h*0.60:
        img_pred = 255 - img_pred
        print('swapping colors')

    res = 0

    for x in range(w):
        for y in range(h):
            diff = np.square(img_true[x, y] - img_pred[x, y])
            res += diff
    return res / (w * h)


if __name__ == '__main__':
    N = 100 ** 2
    iterations = 5
    target = []

    for file in glob.glob(TRAIN_PATH):
        target.append(file)

    target = images2xi(target, N)
    network = HopfieldNetwork(N=N)

    print("TRAINING")
    network.train_pattern(target)
    network.save_network("./hopefield_network.npz")

    print("TESTING")
    network.load_network("./hopefield_network.npz")

    correct = 0
    no_files = 0
    for file in glob.glob(TEST_PATH):
        img_test = images2xi([file], N)
        network.set_initial_neurons_state(np.copy(img_test[:, 0]))
        network.update_neurons(iterations, 'async')

        test_pil = xi_to_PIL(img_test[:, 0], N)
        res_pil = xi_to_PIL(network.S, N)

        true_class = file.split(".png")[0]
        true_class = true_class.split("\\")[1]
        true_class = true_class.split("_")[0]

        prediction_class = predict(res_pil)
        print(f"Prawdziwa: {true_class}, predykcja: {prediction_class}")
        if true_class == prediction_class:
            correct += 1

        # show_images([test_pil, res_pil])
        no_files += 1

    print(f"class predicted correctly: {correct/no_files * 100}")
