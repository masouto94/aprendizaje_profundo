# TP2 explicado para alguien que nunca vio el tema

Astorga · Aprendizaje Profundo, CEIA FIUBA · Cohorte 27

> Este documento no da nada por sabido. Cada palabra técnica se explica la primera vez que aparece. 
> 
> [ASTORGA-GONZALEZ-SOUTO-TP2-Co26.ipynb - Google Drive](https://drive.google.com/file/d/1dtYmyKRkOg293qv1W3axEGxo7pgmswvM/view?usp=sharing)

---

## Abstract

Tenemos datos de un censo de Estados Unidos: unas 45 mil personas con su edad, estudios, trabajo, estado civil, país de nacimiento, horas trabajadas, habilidades y otras características. Para cada persona sabemos si gana más o menos de 50 mil dólares por año. El objetivo es enseñarle a una red neuronal a adivinar ese dato a partir del resto.

La mayor parte del trabajo está en cómo convertir las palabras en números. La red solo entiende números, y varias columnas son texto: la ocupación, el país, la habilidad principal. Hay tres formas de hacer esa conversión y la consigna pedía elegir, para cada columna, la que corresponde. La que la consigna exige usar en el primer modelo se llama embedding. En lugar de que nosotros le asignemos un número a cada categoría, la red aprende sola un pequeño grupo de números (un vector) para cada una, de manera que las categorías que se comportan parecido terminan con números parecidos.

Armamos dos modelos. Uno usa embeddings para las columnas con muchas categorías y una técnica contra la memorización llamada dropout. El otro usa la forma más simple de conversión (una columna por cada categoría) y no tiene dropout. Los dos aciertan más o menos lo mismo: alrededor del 89% de las personas. Pero el de embeddings necesita la mitad de "piezas ajustables", su archivo pesa la mitad y es mucho más estable: si se lo deja entrenar de más, sigue igual de bien, mientras que el otro empieza a memorizar y empeora. Además, al mirar lo que aprendieron los embeddings, vimos que la red ordenó sola las habilidades según cuánto ganan las personas que las tienen.

---

## Contenido teórico

### 1. Clasificar

En el TP1 la red predecía un número (cuánto gasta un cliente). Acá predice una de dos opciones: gana más de 50 mil o no. Eso se llama clasificación binaria. La red devuelve una probabilidad entre 0 y 1, y si pasa de 0,5 decimos que la persona gana más de 50 mil.

### 2. Convertir palabras en números

Hay columnas que ya son números con sentido, como la edad o las horas trabajadas. A esas solo las llevamos a una escala común (estandarizar), para que ninguna pese más que otra solo por tener números más grandes.

Las columnas de texto se pueden convertir de tres formas:

- Ordinal: se numeran las categorías en su orden natural. Sirve cuando el orden existe, como en el nivel educativo: primaria, secundaria, universidad, maestría, doctorado.
- One-hot: una columna por cada categoría, con un 1 en la que corresponde y 0 en las demás. Si la ocupación tiene 14 valores, se convierte en 14 columnas. Sirve cuando no hay orden, pero si hay muchas categorías se generan muchísimas columnas casi siempre vacías, y para la red todas las categorías quedan igual de distintas entre sí.
- Embedding: a cada categoría se le asigna un vector corto, por ejemplo de 5 números, que al principio es aleatorio y la red va ajustando mientras aprende. Las categorías que influyen de forma parecida en el resultado terminan con vectores parecidos. Es lo que se recomienda cuando hay muchas categorías.

En clase se dio una regla práctica: con menos de 10 valores, one-hot; con 10 o más, embedding, con una dimensión inicial igual a la raíz cuadrada de la cantidad de categorías dividida por 2.

$$
\text{dimensión inicial del embedding} = \frac{\sqrt{\text{cantidad de categorías}}}{2} \tag{1}
$$

Por ejemplo, la habilidad principal tiene 80 valores. La raíz de 80 es 8,9, y la mitad es 4,5, que redondeamos a 5. Cada habilidad queda descripta por 5 números.

### 3. La red

Es como la del TP1: capas de neuronas, donde cada capa combina lo que recibe y lo pasa a la siguiente. La nuestra tiene tres capas ocultas, de 128, 64 y 32 neuronas. Las capas de embedding van antes: traducen cada categoría a su vector y lo juntan con el resto de los datos.

Dentro de cada neurona se aplica una función de activación. Usamos LeakyReLU, que deja pasar los valores positivos tal cual y achica mucho los negativos sin anularlos del todo. Fue la recomendación de la clase para modelos con embeddings.

### 4. Cómo se mide el error al entrenar

Para clasificar entre dos opciones se usa una función de error llamada Binary CrossEntropy. Castiga mucho a la red cuando está muy segura y se equivoca: decir "99% seguro de que gana más de 50 mil" cuando en realidad no, cuesta mucho más que decir "55%".

### 5. Memorizar en vez de aprender, y el dropout

Una red con muchas piezas ajustables puede memorizar a las personas con las que entrena en lugar de aprender patrones generales. Se nota porque acierta cada vez más con esas personas y cada vez menos con personas nuevas. Eso se llama sobreajuste.

El dropout es una técnica para evitarlo: mientras entrena, apaga al azar una parte de las neuronas en cada paso (en nuestro caso, el 30%). La red no puede depender de ninguna neurona en particular y tiene que aprender cosas más generales. Cuando se usa el modelo de verdad, el dropout se desactiva y trabajan todas las neuronas.

Otra herramienta es el early stopping (parada temprana): guardar la red en el momento en que mejor le iba con datos que no usa para aprender, y quedarse con esa versión.

### 6. Cómo se mide si el modelo anda bien

- Accuracy: el porcentaje de personas bien clasificadas. Acá engaña un poco: como solo el 25% gana más de 50 mil, un modelo que siempre diga "no" ya acierta el 75%.
- Precision y recall: para la clase "más de 50 mil", el recall dice de cada 100 personas que realmente ganan más, a cuántas detectó el modelo. La precision dice de cada 100 personas que el modelo marcó como "más de 50 mil", cuántas lo eran de verdad.
- F1: combina precision y recall en un solo número.
- F1 macro: el promedio del F1 de las dos clases. Le da el mismo peso a la clase chica que a la grande, por eso es la métrica principal que pide la consigna.
- Matriz de confusión: una tabla de 2 × 2 que cuenta los aciertos y los errores de cada tipo. La versión normalizada por fila muestra lo mismo en porcentajes de cada clase real.

### 7. Parámetros y peso del modelo

Los parámetros son los números que la red ajusta al aprender. Mientras menos tenga para lograr lo mismo, mejor: ocupa menos memoria, el archivo pesa menos y tiene menos capacidad de memorizar. Por eso la consigna pide comparar también esto, y no solo los aciertos.

---

## Lo que hicimos

### a) Entender los datos y decidir cómo convertir cada columna

Primero revisamos los datos. Son 30.162 personas para entrenar y 15.060 para validar, sin datos faltantes. El 25% gana más de 50 mil. Hay algunas filas repetidas, algo normal en un censo sin nombres: dos personas distintas pueden tener exactamente los mismos datos.

Las columnas numéricas cuentan cosas razonables: los que ganan más son mayores (44 años contra 37 en promedio) y trabajan más horas. Las ganancias de capital son 0 para casi todos, pero cuando aparecen son enormes, con un tope de 99.999 que siempre corresponde a gente de ingresos altos. A esas columnas les aplicamos un logaritmo antes de escalarlas, para achicar esos valores extremos.

Después contamos cuántos valores distintos tiene cada columna de texto. Cuatro tienen 10 o más: la habilidad principal (80), el país (41), el nivel educativo (16) y la ocupación (14).

Para cada columna miramos cómo cambia el porcentaje de personas que ganan más de 50 mil entre sus categorías, y con eso decidimos:

- El nivel educativo tiene muchos valores, pero también un orden real, y a más estudios, más ingreso (de 0% en preescolar a 75% en doctorado). Lo convertimos con ordinal: un solo número que respeta ese orden.
- Estado civil, relación en el hogar, tipo de empleador y raza tienen pocos valores y no tienen orden. Los convertimos con one-hot.
- El sexo, con dos valores, pasa a 0 y 1.
- Ocupación, país y habilidad principal tienen muchos valores y no tienen orden. Para estas usamos embeddings, de 2, 4 y 5 números respectivamente.

La habilidad principal resultó ser la columna que más ayuda a predecir el ingreso: va de 3% en cocina básica a 97% en apoyo a decisiones ejecutivas.

### b) El modelo con embeddings

Separamos un 15% de los datos de entrenamiento para tomar decisiones (cuándo cortar el entrenamiento, qué opción elegir), de manera que el set de validación del enunciado quede limpio para medir.

Antes de fijar la configuración probamos varias opciones, cada una 3 veces con arranques distintos, para no dejarnos engañar por la suerte.

- Cómo arrancan los embeddings. Fue lo que más cambió el resultado. Si los vectores arrancan con valores aleatorios grandes, que es lo que hace PyTorch por defecto, la red los usa como etiquetas fijas y casi no los ajusta. Si arrancan con valores chicos, la red los va acomodando. Con valores chicos el F1 macro subió de 0,840 a 0,848.
- Cuántos números por categoría. Duplicar las dimensiones no mejoró nada, así que mantuvimos las de la regla de la clase.
- Cuánto dropout. Entre 20% y 40% dio lo mismo, y nos quedamos con 30%.
- Darle más peso a la clase chica. No mejoró el F1 macro, así que no lo usamos.

El modelo final clasifica bien al 89,2% de las personas, con F1 macro de 0,851. Detecta 3 de cada 4 personas que ganan más de 50 mil, y cuando dice que alguien gana más de 50 mil, acierta el 80% de las veces. En las curvas, entrenamiento y validación avanzan juntos y se separan poco.

También miramos qué aprendieron los embeddings, dibujando los vectores de cada ocupación y de cada habilidad. Quedaron ordenados: las habilidades de ingresos altos de un lado y las de ingresos bajos del otro, con una correlación de 0,97 con el porcentaje real de cada una. La red nunca recibió esos porcentajes como dato: los dedujo de los ejemplos.

### c) El modelo sin embeddings

Es la misma red, pero las tres columnas con muchas categorías pasan a one-hot y se saca el dropout. Ahora la red recibe 166 números por persona en lugar de 42.

Este modelo llega a su mejor punto en la época 4 y desde ahí empeora con las personas nuevas, mientras sigue mejorando con las de entrenamiento: está memorizando. Si nos quedamos con la época 4, sus resultados son casi iguales a los del otro modelo (88,7% de aciertos, F1 macro 0,846). Si lo dejamos entrenar las 40 épocas, cae a 0,813.

### d) La comparación

|                                           | Con embeddings + dropout | One-hot sin dropout |
| ----------------------------------------- | ------------------------ | ------------------- |
| Personas bien clasificadas                | 89,2%                    | 88,7%               |
| F1 macro                                  | 0,851                    | 0,846               |
| Números que recibe la red por persona     | 42                       | 166                 |
| Piezas ajustables (parámetros)            | 16.476                   | 31.745              |
| Peso del archivo                          | 68 KB                    | 127 KB              |
| F1 macro si se entrena de más (40 épocas) | 0,852                    | 0,813               |

Para saber qué parte de la diferencia venía de los embeddings y qué parte del dropout, probamos también las dos combinaciones intermedias. Cortando a tiempo, las cuatro variantes aciertan prácticamente lo mismo. Dejándolas entrenar de más, los embeddings protegen más contra la memorización que el dropout, y las dos cosas juntas son lo mejor.

Conclusión: en este problema, los embeddings no clasifican mucho mejor, pero logran lo mismo con un modelo de la mitad del tamaño, mucho más estable y que aprende relaciones con sentido entre las categorías.

---

## Auditoría posterior

### Qué controlamos

| Qué revisamos                                                          | Para qué                                   | Resultado                                               |
| ---------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| Que todo el preprocesamiento se ajuste solo con datos de entrenamiento | Que no se filtre información de validación | Escalas y vocabularios salen de train                   |
| Decisiones tomadas con un 15% separado de train                        | Que validación quede limpia para medir     | Validación no se usó para elegir nada                   |
| 3 semillas por experimento y una regla de decisión                     | No confundir suerte con mejora             | Solo la inicialización superó el ruido                  |
| Categorías que no aparecen en entrenamiento                            | Que el modelo no falle con datos nuevos    | Índice reservado para "desconocido"                     |
| Diseño 2 × 2 (embeddings sí/no, dropout sí/no)                         | Separar el efecto de cada cambio           | Los embeddings frenan más el sobreajuste que el dropout |
| Dibujar los embeddings aprendidos                                      | Comprobar que aprendieron algo con sentido | Correlación de 0,97 con el ingreso                      |

### Preguntas que nos podrían hacer

#### ¿Por qué el nivel educativo va con ordinal si tiene 16 valores?

Porque tiene un orden real, y los datos lo confirman: a más estudios, más ingreso. Un número que respete ese orden le da a la red esa información directamente. Un embedding tendría que redescubrirla.

#### ¿Por qué Binary y no Categorical CrossEntropy?

Porque hay dos clases y una sola probabilidad alcanza para describir el problema: si la probabilidad de ganar más de 50 mil es 0,8, la de no ganarlos es 0,2. Categorical necesitaría dos salidas para decir lo mismo.

Si aciertan casi igual, ¿cuál es la ventaja de los embeddings?
La mitad de parámetros y de peso, mucha menos memorización y relaciones aprendidas entre categorías. Eso pesa más cuando hay muchas más categorías o menos datos.

#### ¿Por qué el modelo sin embeddings memoriza tanto?

Porque cada una de sus 135 columnas nuevas tiene 128 pesos propios que se aprenden solo con las personas de esa categoría. En categorías con pocas personas, la red termina memorizándolas.

#### ¿Por qué cambió tanto la inicialización?

Porque con valores iniciales grandes los vectores casi no se mueven durante el entrenamiento y quedan como etiquetas aleatorias. Con valores chicos, la red los puede acomodar según lo que sirve para predecir.

### Fuentes

- Material de la materia, clase 4 (dropout y embeddings): https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/aprendizaje_profundo
- Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*: https://jmlr.org/papers/v15/srivastava14a.html
- Guo y Berkhahn (2016), *Entity Embeddings of Categorical Variables*: https://arxiv.org/abs/1604.06737
- Kingma y Ba (2015), *Adam: A Method for Stochastic Optimization*: https://arxiv.org/abs/1412.6980
