# satcomp

Plataforma local para correr competencias entre SAT solvers sobre instancias CNF.

## Requisitos

- Python 3.11+

## Instalacion

```bash
pip install -e .
```

## Inicializar

```bash
satcomp init
```

Esto crea carpetas y un `config.yaml` base.

## Agregar solvers

Coloca archivos YAML/JSON en `./solvers/`.

Ejemplo (`./solvers/minisat.yaml`):

```yaml
name: minisat
version: "2.2.0"
bin: "./solvers/bin/minisat"
command_template: "{bin} -cpu-lim={timeout} {cnf}"
supports_seed: false
default_threads: 1
```

Placeholders disponibles: `{cnf}`, `{seed}`, `{timeout}`, `{mem_mb}`, `{threads}`, `{bin}`.
Si hay espacios en rutas, usa comillas en `command_template`.
En Windows, si tu solver corre en WSL (por ejemplo `bin: "wsl kissat"`), `{cnf}` se convierte automaticamente a una ruta tipo `/mnt/<drive>/...`.

Valida:

```bash
satcomp solvers validate
```

## Indexar instancias

Coloca CNF en `./benchmarks/` y corre:

```bash
satcomp instances index
```

## Crear y correr un run

```bash
satcomp run create --name baseline --solvers minisat,cadical --timeout 60 --mem-mb 2048
satcomp run start 1 --jobs 4
```

Reanudar: `satcomp run start 1` detecta pendientes y continua.

## Reportes y dashboard

```bash
satcomp report 1
satcomp serve --host 127.0.0.1 --port 8000
```

Luego abre `http://127.0.0.1:8000`. Para ver progreso en vivo: `http://127.0.0.1:8000/runs/<id>/live`.

## Exportar / Importar

```bash
satcomp export 1 --format jsonl
satcomp import export_run_1.jsonl
```

## Estructura

```
./benchmarks/    # CNF
./solvers/       # YAML/JSON
./runs/          # logs
./reports/       # reportes
./data/results.db
```

## Tests

```bash
pytest
```
