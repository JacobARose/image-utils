# Python-boilerplate

TODO: short introduction of this project. The objective or the motivation
behind this project.

## Getting Started

1. Installation process
1. Software dependencies
1. Latest releases
1. API references


## Documentation

### 1. Rebuild html api docs
```
export PORT=9797 # Optionally override 
make docs PORT=PORT
```

### 2. Host api docs on local server
```
export PORT=9797 # Optionally override default port of 9797
make docs-server PORT=$PORT
```
then, on local shell, build an ssh tunnel
```
# e.g.
export SERVER_IP="serrep3.services.brown.edu"

ssh -N -L PORT:localhost:PORT $(USER)@$(SERVER_IP)
```





## Changelog

Please read the [changelog](CHANGELOG.md) to check any notable changes of this project.

## Contributing

Please read the [contribution guidelines](CONTRIBUTING.md) before starting work on a pull request.
