## Utiliser le lidar

```sh
 pip install rplidar
```

### trouver le port

```sh
ls /dev/tty* | grep tty.usb #pour mac
ls /dev/tty* | grep ttyUSB #pour linux
```

### Se connecter au robot

- se connecter au wifi miniNox
- se connecter en ssh

```sh
ssh root@172.24.1.1 #mdp root
```

- (transférer les fichiers en ftp ou créer le fichier sur la raspberry)

- lancer le script
