# Guía de uso del ecosistema internado-investigacion

## Paso a paso para iniciar y trabajar

1. **Clona el repositorio en cualquier computadora:**
   ```bash
   git clone https://github.com/DrCiroSmith/internado-investigacion.git
   ```

2. **Copia `.env.example` a `.env` y configura tus credenciales:**
   - Edita el archivo `.env` con tus datos de acceso a MySQL.

3. **Coloca tus archivos Excel (.xlsx) en la carpeta `data/`:**
   - Todos los archivos que quieras explorar, limpiar y analizar deben ir en `internado-investigacion/data/`.
   - Puedes organizar subcarpetas si lo deseas.

4. **Ejecuta el entorno con Docker:**
   ```bash
   docker-compose up --build
   ```

5. **Accede a Jupyter en tu navegador:**
   - Ingresa a `localhost:8888`.
   - Los notebooks están en la carpeta `notebooks/`.
   - Desde Jupyter puedes cargar, explorar y limpiar tus archivos Excel usando pandas y openpyxl.
   - Para separar sheets de interés, usa pandas:  
     ```python
     import pandas as pd
     xls = pd.ExcelFile('/data/tu_archivo.xlsx')
     df_sheet1 = pd.read_excel(xls, 'Sheet1')
     ```

6. **Carga los datos limpios a MySQL:**
   - Usa las credenciales configuradas en `.env`.
   - Ejemplo de conexión en notebook:
     ```python
     import mysql.connector
     conn = mysql.connector.connect(
         host='mysql',
         user='DrNemo',
         password='tu_password',
         database='PredictorDeExodoncias'
     )
     ```

7. **Sincroniza tus notebooks y código con GitHub para portabilidad total:**
   - Haz commit y push de tus notebooks y scripts en la carpeta `notebooks/`.

## Recomendaciones

- No subas archivos de la carpeta `data/` ni bases de datos a GitHub.
- Realiza backups cifrados de tus datos sensibles.
- Documenta tu flujo y hallazgos en Notion o en archivos Markdown dentro del repo.
