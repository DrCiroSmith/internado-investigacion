# internado-investigacion

Ecosistema portátil y seguro para análisis clínico con Jupyter, Python 3.12, MySQL 8.0 y Docker.

## Estructura del proyecto

- `notebooks/` : Jupyter Notebooks para exploración, limpieza y análisis.
- `data/` : Archivos Excel/CSV (no subir a GitHub).
- `mysql/` : Persistencia de base de datos MySQL (no subir a GitHub).
- `forms/` : Formularios PDF/imágenes (no subir a GitHub).
- `docker/` : Dockerfile y configuración de entorno.
- `docker-compose.yml` : Orquestación de servicios.
- `.env.example` : Variables de entorno de ejemplo.
- `.gitignore` : Exclusión de carpetas sensibles.

## Seguridad

- Los datos clínicos y formularios nunca se suben al repositorio.
- Se recomienda cifrar carpetas sensibles con Cryptomator o gocryptfs.
- Exporta la base MySQL periódicamente y encripta con gpg si es necesario.

## Uso

1. Clona el repositorio.
2. Copia `.env.example` a `.env` y configura tus credenciales.
3. Ejecuta `docker-compose up --build`.
4. Accede a Jupyter en `localhost:8888`.
5. Trabaja en tus notebooks y sincroniza el código con GitHub.
