# TP1 · Pregunta 1: GD vs Adam en una neurona con tanh

**Víctor Astorga** · Aprendizaje Profundo, CEIA FIUBA · Cohorte 27

---

## Abstract

La consigna pedía comparar tres formas de entrenar la red neuronal más chica que existe: una sola neurona con dos parámetros, w y b. Los tres métodos eran Gradiente Descendente clásico, Adam usando todos los datos en cada paso y Adam usando grupos de 32 datos. Cada uno se probaba con tres tamaños de paso (0.1, 0.01 y 0.001) durante 100 épocas, siempre desde el mismo punto de partida.

Inventé un dataset con forma de "S" ondulada. A propósito, tiene partes que la neurona no puede copiar. Eso me dejó un error mínimo conocido (0.0343), que usé como línea de llegada para saber si un método realmente había convergido o solo se había quedado cerca.

Lo principal que encontré es que no hay un ganador fijo: depende del tamaño de paso. Con 0.1, GD termina perfecto. Con 0.01, solo Adam por grupos llega. Con 0.001, no llega ninguno en 100 épocas. La razón está en cómo es el terreno del error. Tiene mesetas, zonas casi planas donde la pendiente es muy chica. GD avanza en proporción a la pendiente, así que en las mesetas se queda casi quieto. Adam avanza más o menos lo mismo en cada paso, por eso escapa de las mesetas, pero su inercia a veces lo hace pasarse del mínimo. Lo comprobé apagando cada pieza de Adam por separado.

También vi que Adam por grupos no gana porque cada paso sea mejor, sino porque da 16 pasos donde los otros dan uno, con el mismo trabajo de cómputo. Para no sacar conclusiones de un solo arranque, repetí todo desde 500 puntos de partida distintos. Ahí el método más robusto fue Adam por grupos, después Adam con todos los datos y al final GD.

---

## Contenido teórico

### La neurona

Una neurona es una fórmula que recibe un número x y devuelve una predicción. Primero arma una recta con dos parámetros: w, que define la inclinación, y b, que la desplaza. Después pasa el resultado por la tangente hiperbólica (tanh), una función con forma de "S" que aplasta cualquier número al rango entre -1 y 1.

$$
\hat{y} = \tanh(w\,x + b) \tag{1}
$$

Si se cambia w, la "S" se hace más empinada o más suave. Si w es negativo, la "S" queda dada vuelta. Si se cambia b, la "S" se corre hacia un costado. Entrenar es encontrar automáticamente los valores de w y b con los que la curva se parece más a los datos.

La tanh tiene una característica que termina siendo clave en todo el trabajo. Cuando recibe números muy grandes o muy negativos, su salida queda pegada a 1 o a -1 y casi no cambia. En ese caso se dice que está saturada. Una neurona saturada puede estar equivocándose mucho y, aun así, mover un poco w o b casi no cambia su salida.

### Medir el error

Para saber qué tan bien anda el modelo se usa el error cuadrático medio (MSE). Para cada dato se calcula la diferencia entre lo que predijo el modelo y el valor real, se la eleva al cuadrado y al final se promedian todas.

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}_i - y_i\right)^2 \tag{2}
$$

Elevar al cuadrado evita que los errores para arriba y para abajo se cancelen entre sí, y hace que los errores grandes pesen más. Entrenar es buscar el w y el b que hacen este número lo más chico posible.

### El terreno del error

Una forma útil de imaginarlo es como un mapa. Un eje es w, el otro es b, y la altura del terreno en cada punto es el error. Buscar los mejores parámetros es buscar el punto más bajo de ese mapa. El profesor lo comparó con caminar con los ojos vendados por una montaña, tanteando la inclinación del suelo con los pies.

Si el terreno fuera un pozo perfecto (convexo), cualquier camino cuesta abajo llevaría al fondo. El de este problema no es así. Tiene zonas con forma de silla de montar y, sobre todo, mesetas: llanos altos donde el error es grande pero el suelo es casi plano. Las mesetas aparecen justamente donde la tanh está saturada.

### La pendiente (gradiente)

El gradiente indica para dónde sube el error y qué tan empinado está, separado para w y para b. Caminar en sentido contrario es bajar. Se calcula con la regla de la cadena, desde el error hacia atrás: cuánto cambia el error con la predicción, cuánto cambia la predicción con la tanh y cuánto cambia la tanh con w o con b.

En esa cadena aparece la derivada de la tanh, que en zona saturada es casi cero. Por eso, en una meseta, la pendiente también es casi cero aunque el modelo se equivoque muchísimo.

### Tamaño de paso, épocas y lotes

El learning rate define cuánto se avanza en cada paso. Si es muy chico, no se llega nunca. Si es muy grande, se puede saltar por encima del mínimo.

Una época es una pasada completa por todos los datos. Cuando se usan todos los datos para cada paso (full-batch), hay un paso por época. Cuando se usan grupos (mini-batch, en mi caso de 32 datos), hay un paso por grupo: con 512 datos son 16 pasos por época. Cada grupo da una estimación de la pendiente que en promedio apunta bien pero tiene algo de ruido, porque mira solo una parte de los datos.

### Gradiente Descendente

Es la regla más simple: el nuevo valor de cada parámetro es el valor actual menos el learning rate multiplicado por la pendiente.

$$
w \leftarrow w - \text{lr}\cdot\frac{\partial J}{\partial w}
\qquad\qquad
b \leftarrow b - \text{lr}\cdot\frac{\partial J}{\partial b} \tag{3}
$$

Lo importante es que el tamaño del paso es proporcional a la pendiente. Donde el terreno es empinado avanza rápido. Donde es plano, casi no se mueve.

### Adam

Adam le agrega a cada parámetro dos memorias. La primera, m, es un promedio de las pendientes recientes. Funciona como inercia: si venía bajando para un lado, tiende a seguir para ese lado. La segunda, v, es un promedio de las pendientes al cuadrado y le da una idea de qué tan grandes suelen ser las pendientes en esa zona. Recuerda mucho más tiempo que la primera, del orden de mil pasos contra diez. El paso se arma dividiendo una memoria por la otra.

$$
\theta \leftarrow \theta - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} \tag{4}
$$

(Los sombreritos indican una corrección que se hace al principio, porque las dos memorias arrancan en cero.)

La idea clave es que ese cociente queda cerca de 1, o de -1, siempre que la pendiente sea consistente, sin importar si es diminuta o enorme. Entonces Adam avanza más o menos el learning rate en cada paso y en cada parámetro por separado. En una meseta sigue avanzando, mientras que GD casi se detiene. A cambio, la inercia lo puede hacer pasarse, y la memoria larga lo puede dejar dando pasos chicos cuando ya no hace falta.

### Dos números de referencia

El primero es el mejor error posible para esta neurona, que llamé J*. Como el modelo tiene solo dos parámetros, lo pude encontrar probando unas 154.000 combinaciones y refinando la mejor. Dio 0.0343. Ningún optimizador puede bajar de ese valor.

El segundo es el R², que compara el modelo con la estrategia más tonta posible: predecir siempre el promedio de los datos. Un R² de 1 es perfecto, uno de 0 equivale a predecir el promedio y uno negativo es peor que eso. Para esta neurona, el máximo alcanzable es 0.897.

---

## Lo que hice

### Preparación

Lo primero fue inventar los datos. Probé varias combinaciones y me quedé con una suma de tres partes: un seno, un polinomio de grado tres y una campana exponencial corrida hacia la derecha. Tomé 512 puntos entre -3 y 3 y les sumé un poco de ruido aleatorio. La forma general es una "S" que la neurona puede imitar, pero hay un tramo que sube y después baja, y una "S" no puede hacer eso. Lo elegí así a propósito, para que quedara un error mínimo que no fuera culpa de ningún optimizador. La campana la corrí para que el parámetro b también importara. Usé 512 puntos porque se dividen justo en 16 grupos de 32.

Después programé el modelo y la pendiente a mano. Antes de usarla, la comparé con una aproximación numérica: muevo cada parámetro un poquito hacia cada lado y veo cuánto cambia el error. Coincidieron hasta la décima cifra. Me pareció necesario porque un error de signo en una derivada puede dar curvas que se ven razonables y estar mal igual.

Por último calculé J*. De ese error mínimo, cerca del 70% es culpa de la neurona (no puede dibujar el tramo que sube y baja) y el resto es básicamente ruido. Con eso quedó claro que converger en este problema significa llegar a 0.0343, no a cero.

Para el punto de partida usé la semilla 35. La elegí después de probar otras. La 42, por ejemplo, arrancaba casi encima del mínimo, los tres métodos llegaban igual y no había nada para comparar. La 35 arranca con la "S" dada vuelta y saturada, o sea, en una meseta. Es un caso difícil, y por eso después repetí todo con 500 arranques distintos.

### 1a · Gradiente Descendente

En el gráfico de 1a están las tres curvas de error superpuestas. Con 0.1 la curva tiene forma de escalón: durante unas 40 épocas casi no baja, entre la 40 y la 50 cae de golpe y en la 64 ya está pegada a J*. Lo que pasa es que al principio la pendiente es chiquita porque el modelo está saturado. Cuando w se acerca a cero, el modelo sale de la saturación, la pendiente crece y los pasos se agrandan. En ningún momento el error sube.

Con 0.01 y 0.001 recorre el mismo camino, pero diez y cien veces más lento, y en 100 épocas no sale de la meseta. No está atrapado en un mínimo falso: simplemente no tiene pendiente que lo empuje.

### 1b · Adam con todos los datos y con grupos

En los dos paneles de 1b se ve que Adam con todos los datos y 0.1 arranca mucho más rápido que GD (en la época 13 ya había bajado bastante, mientras GD seguía casi igual). Pero después se pasa del mínimo y termina con un error tres veces mayor que J*. Con 0.01 avanza de forma pareja sin llegar, y con 0.001 casi no se mueve.

Adam con grupos y 0.1 es lo más rápido de todo el trabajo: en la época 2 ya había hecho el 90% de la bajada. Pero después no se queda quieto y queda temblando alrededor del mínimo. Con 0.01 dio el mejor resultado: llega en unas 12 a 14 épocas y se queda en J*. Con 0.001 todavía estaba bajando al llegar a la época 100.

Hay una cuenta simple que explica casi todo. Si Adam avanza más o menos el learning rate por paso y w tiene que recorrer 2.34 para llegar al mínimo, hacen falta como mínimo 23 pasos con 0.1, 234 con 0.01 y 2340 con 0.001. Con todos los datos hay 100 pasos; con grupos, 1600. Con eso ya se ve quién puede llegar y quién no.

Dos detalles de implementación que tuve en cuenta. El contador de pasos de Adam no se reinicia en cada época, porque si no la corrección del principio vuelve a inflar los pasos. Y el error de cada época lo mido siempre sobre todos los datos, para poder comparar con GD.

### 1c · Comparación

Hice tres paneles, uno por cada learning rate, con los tres métodos en distintos colores y tipos de línea. También armé una tabla con el error final, cuánto quedó por encima de J*, en qué época logró el 90% de la bajada, desde qué época quedó estable, cuántas veces empeoró, cuántos pasos dio y cuánto tardó.

En los nueve gráficos de datos contra curva aprendida, los casos que convergen quedan encima de la mejor "S" posible y fallan todos en lo mismo: no siguen el tramo que sube y baja. Los que no salieron de la meseta muestran una "S" invertida, mucho peor que predecir el promedio.

Después grafiqué el error contra la cantidad de pasos en vez de contra las épocas. Ahí la ventaja de Adam con grupos casi desaparece: con 100 pasos cada uno, queda casi igual que Adam con todos los datos. O sea, un paso calculado con 32 datos vale casi lo mismo que uno calculado con 512. Adam con grupos gana porque, con el mismo trabajo por época, da 16 pasos.

Por último, los 500 arranques. Ahí descubrí que la semilla 35 está entre el 1% más difícil, así que tuve que corregir una conclusión. Con 0.1, en general los tres llegan y GD rinde parecido a Adam con todos los datos. Que Adam haya salido peor que GD era cosa de esa semilla. Con 0.01, Adam con grupos llegó en el 100% de los arranques, Adam con todos los datos en el 31% y GD en el 5%. Con 0.001, solo Adam con grupos se acerca, en un 52%.

Mi conclusión es que GD es el más prolijo cuando el learning rate es el adecuado, pero es frágil. Adam no depende de la pendiente, pero sí del learning rate. Adam con grupos es el más robusto. Para este problema, elegiría Adam con grupos y 0.01.

### 1d · Trayectorias en 3D

Antes de dibujar los caminos hice dos mapas del terreno. El primero muestra que el 97% de la zona no es convexa: solo hay una isla con forma de pozo alrededor del mínimo. El segundo muestra las mesetas, en los cuatro costados, y un canal empinado en el medio que lleva al mínimo.

El learning rate para el 3D lo elegí midiendo cuánto se separaban los tres caminos en cada caso. El de mayor separación fue 0.01. Con ese valor:

- GD apunta bien, pero en 100 épocas recorre apenas 0.16 y casi no se ve.
- Adam con todos los datos va en línea recta a 45 grados, porque mueve los dos parámetros lo mismo aunque una pendiente sea 6.5 veces más grande que la otra. Sale de la meseta, pero no en la dirección ideal.
- Adam con grupos hace la misma diagonal, encuentra el canal, gira y llega al mínimo. Una vez ahí, el ruido de los grupos se cancela en su memoria y los pasos se achican solos.

Con 0.1, que se ve en la vista desde arriba, Adam con todos los datos da un rodeo enorme y se pasa hasta la meseta del otro lado. Desde ahí vuelve muy despacio, porque su memoria larga todavía recuerda las pendientes fuertes del canal. Para comprobarlo repetí esa corrida cambiando una pieza por vez. Sin la inercia, no se pasa y llega en 25 épocas, antes que GD. Con la memoria larga más corta, se pasa igual pero vuelve rápido. Con 300 épocas, la versión original también llega. Es decir, la inercia causa que se pase y la memoria larga causa que la vuelta sea lenta.

Al final dejé una animación que muestra cómo avanza cada uno época por época.

---

## Auditoría posterior

Cuando terminé, revisé el trabajo como si fuera de otra persona. La idea era encontrar errores antes de mostrarlo y anticipar las preguntas que me podrían hacer.

### Qué verifiqué

| Qué revisé | Para descartar qué | Resultado |
|---|---|---|
| Pendiente programada a mano contra aproximación numérica | Errores en las derivadas | Coinciden hasta la décima cifra |
| Lo mismo para las segundas derivadas | Errores en el análisis del terreno | Coinciden |
| Versión rápida (500 arranques en paralelo) contra versión paso a paso | Que el estudio de robustez use otro algoritmo | Diferencia cero |
| 500 arranques distintos | Conclusiones que dependan de una semilla | Corregí una conclusión |
| Apagar piezas de Adam una por vez | Explicaciones sin evidencia | Cada mecanismo confirmado |
| Mis 9 corridas contra GD y Adam de PyTorch (fuera del notebook, porque la consigna no permite usarlo) | Errores en mi implementación | Iguales hasta 15 decimales |
| J* calculado con otro método (L-BFGS de PyTorch) | Error en la línea de llegada | Mismo valor |

### Preguntas que me podrían hacer

**¿Elegir la semilla 35 no es acomodar el resultado?** La elegí porque con un arranque fácil no había nada que comparar. Lo dejé escrito en el notebook y repetí todo con 500 semillas para separar lo que se generaliza de lo que no.

**¿Por qué GD con 0.01 no aprende? ¿Hay un mínimo local?** No hay mínimo local: el terreno tiene uno solo. Está en una meseta, donde la pendiente es casi cero. Con unas 500 épocas llegaría.

**Si Adam es mejor, ¿por qué con 0.1 perdió contra GD?** En esa semilla, la inercia lo hizo pasarse y la memoria larga lo frenó a la vuelta. La prueba de apagar piezas lo muestra. Con los 500 arranques, con 0.1 rinden parecido.

**¿No es injusto comparar con Adam por grupos, que da 16 veces más pasos?** Por época, los tres procesan los mismos 512 datos. Igual agregué la comparación por pasos, y ahí queda claro que la ventaja viene de dar más pasos con el mismo trabajo.

**¿Por qué Adam por grupos tiembla con 0.1?** Cada grupo da una pendiente con ruido, y con pasos grandes ese ruido lo mantiene dando vueltas alrededor del mínimo. Con 0.01 el temblor es diez veces menor. Se arregla achicando el learning rate con el tiempo.

**¿Por qué ninguno llega a error cero?** Porque una sola neurona dibuja una "S" que siempre sube o siempre baja, y los datos suben y bajan. Para eso harían falta más neuronas.

### Fuentes

- Kingma y Ba (2015), *Adam: A Method for Stochastic Optimization*, https://arxiv.org/abs/1412.6980
- Goodfellow, Bengio y Courville (2016), *Deep Learning*, capítulo 8, https://www.deeplearningbook.org/
- Notebook de optimizadores de la clase 2: https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/aprendizaje_profundo/blob/ap_2026/CLASE%202/Notebooks/clase2_4_optimizadores.ipynb
