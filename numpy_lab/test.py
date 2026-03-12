# test.py
# Тесты для модуля main.py
import os
import numpy as np

from main import (
    create_vector,
    create_matrix,
    reshape_vector,
    vector_add,
    scalar_multiply,
    elementwise_multiply,
    dot_product,
    matrix_multiply,
    matrix_determinant,
    matrix_inverse,
    solve_linear_system,
    load_dataset,
    statistical_analysis,
    normalize_data,
    plot_histogram,
    plot_heatmap,
    plot_line)


def test_create_vector():
    """Проверяет создание вектора от 0 до 9."""
    v = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix():
    """Проверяет создание матрицы 5×5 со случайными числами [0, 1)."""
    m = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector():
    """Проверяет изменение формы вектора с (10,) на (2, 5)."""
    v = np.arange(10)
    reshaped = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add():
    """Проверяет поэлементное сложение двух векторов."""
    assert np.array_equal(
        vector_add(np.array([1,2,3]), np.array([4,5,6])),
        np.array([5,7,9])
    )
    assert np.array_equal(
        vector_add(np.array([0,1]), np.array([1,1])),
        np.array([1,2])
    )


def test_scalar_multiply():
    """Проверяет умножение вектора на скаляр."""
    assert np.array_equal(
        scalar_multiply(np.array([1,2,3]), 2),
        np.array([2,4,6])
    )


def test_elementwise_multiply():
    """Проверяет поэлементное умножение двух векторов."""
    assert np.array_equal(
        elementwise_multiply(np.array([1,2,3]), np.array([4,5,6])),
        np.array([4,10,18])
    )


def test_dot_product():
    """Проверяет скалярное произведение двух векторов."""
    assert dot_product(np.array([1,2,3]), np.array([4,5,6])) == 32
    assert dot_product(np.array([2,0]), np.array([3,5])) == 6


def test_matrix_multiply():
    """Проверяет умножение двух матриц."""
    A = np.array([[1,2],[3,4]])
    B = np.array([[2,0],[1,2]])
    assert np.array_equal(matrix_multiply(A,B), A @ B)


def test_matrix_determinant():
    """Проверяет вычисление определителя матрицы 2×2."""
    A = np.array([[1,2],[3,4]])
    assert round(matrix_determinant(A), 5) == -2.0


def test_matrix_inverse():
    """Проверяет вычисление обратной матрицы (A × A⁻¹ = E)."""
    A = np.array([[1,2],[3,4]])
    invA = matrix_inverse(A)
    assert np.allclose(A @ invA, np.eye(2))


def test_solve_linear_system():
    """Проверяет решение системы линейных уравнений Ax = b."""
    A = np.array([[2,1],[1,3]])
    b = np.array([1,2])
    x = solve_linear_system(A, b)
    assert np.allclose(A @ x, b)


def test_load_dataset():
    """Проверяет загрузку CSV файла и возврат numpy массива."""
    test_data = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    try:
        data = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78, 81, 90])
    finally:
        os.remove("test_data.csv")


def test_statistical_analysis():
    """Проверяет расчёт статистики: mean, min, max."""
    data = np.array([10, 20, 30])
    result = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization():
    """Проверяет Min-Max нормализацию данных в диапазон [0, 1]."""
    data = np.array([0, 5, 10])
    norm = normalize_data(data)
    assert np.allclose(norm, np.array([0, 0.5, 1]))


def test_plot_histogram():
    """Проверяет, что функция построения гистограммы не вызывает ошибок."""
    data = np.array([1, 2, 3, 4, 5])
    plot_histogram(data)


def test_plot_histogram_creates_file():
    """Проверяет, что гистограмма сохраняется в файл plots/math_grades_histogram.png."""
    data = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
    expected_path = 'plots/math_grades_histogram.png'

    if os.path.exists(expected_path):
        os.remove(expected_path)

    plot_histogram(data)

    assert os.path.exists(expected_path), f"Файл {expected_path} не был создан!"
    assert os.path.getsize(expected_path) > 0, "Создан пустой файл"


def test_plot_heatmap():
    """Проверяет, что функция построения тепловой карты не вызывает ошибок."""
    matrix = np.array([[1, 0.5], [0.5, 1]])
    plot_heatmap(matrix)


def test_plot_heatmap_creates_file():
    """Проверяет, что тепловая карта сохраняется в файл plots/correlation_heatmap.png."""
    matrix = np.array([[1, 0.5, 0.3],
                       [0.5, 1, 0.7],
                       [0.3, 0.7, 1]])
    expected_path = 'plots/correlation_heatmap.png'

    if os.path.exists(expected_path):
        os.remove(expected_path)

    plot_heatmap(matrix)

    assert os.path.exists(expected_path), f"Файл {expected_path} не был создан"
    assert os.path.getsize(expected_path) > 0, "Создан пустой файл"


def test_plot_line():
    """Проверяет, что функция построения линейного графика не вызывает ошибок."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    plot_line(x, y)


def test_plot_line_creates_file():
    """Проверяет, что линейный график сохраняется в файл plots/student_grades_line.png."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([78, 82, 85, 90, 95])
    expected_path = 'plots/student_grades_line.png'

    os.makedirs('plots', exist_ok=True)

    if os.path.exists(expected_path):
        os.remove(expected_path)

    plot_line(x, y)

    assert os.path.exists(expected_path), f"Файл {expected_path} не был создан"
    assert os.path.getsize(expected_path) > 0, "Создан пустой файл"

