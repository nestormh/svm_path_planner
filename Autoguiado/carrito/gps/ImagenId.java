package carrito.gps;

import java.io.*;
import java.awt.Canvas;

public class ImagenId implements Serializable {

  static {
    System.loadLibrary("javavlc");
  }

  private double posX = -1;
  private double posY = -1;
  private double angulo = -1;

  private int ancho = -1;
  private int alto = -1;

  private int[] imagen;

  String nombre = "";

  private native void _verImagen(double posX, double posY, double angulo,
      int ancho, int alto, int [] imagen, String nombre);

  public ImagenId(double posX, double posY, double angulo,
                 int ancho, int alto, int[] imagen) {
    this.posX = posX;
    this.posY = posY;
    this.angulo = angulo;
    this.ancho = ancho;
    this.alto = alto;
    this.imagen = imagen;
  }

  public double getAngulo() {
    return angulo;
  }

  public int[] getImagen() {
    return imagen;
  }

  public double getPosX() {
    return posX;
  }

  public double getPosY() {
    return posY;
  }

  public int getAlto() {
    return alto;
  }

  public int getAncho() {
    return ancho;
  }

  public String getNombre() {
    return nombre;
  }

  public void setPosY(double posY) {
    this.posY = posY;
  }

  public void setPosX(double posX) {
    this.posX = posX;
  }

  public void setImagen(int[] imagen) {
    this.imagen = imagen;
  }

  public void setAngulo(double angulo) {
    this.angulo = angulo;
  }

  public void setAlto(int alto) {
    this.alto = alto;
  }

  public void setAncho(int ancho) {
    this.ancho = ancho;
  }

  public void setNombre(String nombre) {
    this.nombre = nombre;
  }

  public void verImagen() {
    _verImagen(posX, posY, angulo, ancho, alto, imagen, nombre);
  }

}
