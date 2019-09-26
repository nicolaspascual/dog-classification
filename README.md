# Comandos importantes

## Subir datos
```bash
make upload
```

## Encolar debug
```bash
make debug-task
```

## Encolar train
```bash
make queue-task
```

# Caso de uso

Cambias codigo y quieres probarlo en debug

_cambias codigo_
```bash
make upload
```
_enqueas a debug_
```bash
make debug-task
```
_si funciona bien_
```bash
make queue-task
```