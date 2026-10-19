# Diccionario de datos — Dataset sucio para prácticas de limpieza (6,000 filas)

Este dataset fue generado artificialmente para prácticas de estandarización de datos, deduplicación y consistencia.

## Columnas
- **record_id**: Identificador del registro (string). Mezcla de enteros con/ sin ceros a la izquierda y algunos alfanuméricos.
- **full_name**: Nombre completo (con y sin acentos, posibles variaciones).
- **email**: Correos en diferentes formatos; ~5% con errores (e.g., `[at]` en lugar de `@`), vacíos intencionales.
- **phone**: Teléfonos en múltiples formatos internacionales (e.g., `+52`, `+1`, con/ sin paréntesis, guiones).
- **date_of_birth**: Fechas en formatos mixtos (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`, `DD-Mon-YYYY`, `DD.MM.YYYY`).
- **signup_date**: Fecha/hora de alta con formatos variados (24h/12h, con y sin hora).
- **last_login**: Último acceso; puede estar vacío. Formatos diferentes.
- **country**: País (México, Estados Unidos, Canada, Brasil, España, Argentina).
- **state_province**: Entidad federativa cuando aplica (México) o vacío para otros países. Puede faltar.
- **city**: Ciudad coherente con el país, pero no necesariamente con el estado.
- **postal_code**: Códigos postales de longitud variable; en EUA a veces con extensión `ZIP+4`. Posibles vacíos.
- **customer_status**: Estado del cliente con variantes (`Activo`, `Inactivo`, `Prueba`, `active`, `inactive`, `trial`, `EN PRUEBA`, `ACTIVO`).
- **plan**: Plan contratado con variantes (`Básico`, `Pro`, `Empresarial`, `basic`, `PRO`, `enterprise`).
- **preferred_language**: Idioma preferido (`es`, `en`, `ES`, `EN`, `Español`, `Inglés`).
- **timezone**: Zona horaria en formatos mixtos (`UTC±X`, `PST`, `CST`, `EST`).
- **height**: Estatura *con unidades mezcladas* (`cm` o `in`). **Ideal para actividad de conversión de unidades**.
- **weight**: Peso *con unidades mezcladas* (`kg` o `lb`). **Ideal para actividad de conversión de unidades**.
- **marketing_consent**: Consentimiento con valores heterogéneos (`Sí/No`, `YES/NO`, `True/False`, `1/0`, `Y/N`).
- **amount_spent**: Monto acumulado en formatos heterogéneos (con/ sin símbolo $, separadores, prefijos `MXN`/`USD`).

## Duplicados
~5% de las filas provienen de clusters de duplicados: mismas personas con ligeras variaciones (teléfono, mayúsculas/minúsculas en email, acentos en nombre, plan/estatus, `record_id` con y sin ceros a la izquierda). Sirve para practicar reglas de *record linkage* y selección del registro canónico.

## Ideas de actividades
1. **Estandarización**: Convertir `height` a **centímetros** y `weight` a **kilogramos**; normalizar fechas a ISO `YYYY-MM-DD` y `amount_spent` a número decimal.
2. **Normalización categórica**: Mapear `customer_status` y `plan` a catálogos únicos (`Activo|Inactivo|Prueba` y `Básico|Pro|Empresarial`).
3. **Corrección de emails**: Detectar y reparar patrones comunes (reemplazar `[at]`→`@`, recortar espacios, a minúsculas, validar con regex).
4. **Deduplicación**: Definir claves de *match* (por ejemplo: email normalizado, nombre + fecha de nacimiento, similitud de nombre con `record_id` normalizado). Mantener un registro canónico y eliminar duplicados.
5. **Consistencia**: Verificar coherencia país/estado/ciudad; validar `postal_code` por país; chequear que `last_login` ≥ `signup_date` cuando existan.
6. **Valores faltantes**: Imputar o descartar según reglas justificadas (por ejemplo, `state_province` fuera de México).

## Nota
Este dataset es **sintético** y no contiene datos reales de personas.