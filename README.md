Известная тестовая задача распознавания отдельных рукописных цифр.

База MNIST содержит 50000 обучающих и 10000 тестовых образцов - для обучения классификатора, и еще 10000 образцов для финального тестирования.

Оригинальная база в хитром бинарном формате. Тут используется перепакованная версия (http://deeplearning.net/data/mnist/mnist.pkl.gz), unpack_db.py распакует ее в директории data в виде отдельных файлов и обрежет большие белые поля, чтобы буквы занимали большую часть изображения.

Есть возможность обучить общий классификатор (train_diff.py) или набор бинарных классификаторов (train_bin.py). Примеры обученных классификаторов лежат в nets.

Костяк сверточной сети lenet5 честно стырен с deeplearning.net, отличие в том, что весь код собран воедино и добавлено несколько плюшек для удобства использования, так что lenet5.py содержит standalone-реализацию lenet5 для любых нужд.

apply.py, apply1.py и apply2.py содержат разные комбинации применения общего и бинарных классификаторов на экзаменационной выборке изображений (data/test). На выходе они генерируют изображение diff_out.png.

Пример diff_out.png, сгенерированный общим классификатором, прилагается.