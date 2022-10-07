#! /bin/sh

mysql -uroot -e "drop database $1"
mysql -uroot -e "create database $1"