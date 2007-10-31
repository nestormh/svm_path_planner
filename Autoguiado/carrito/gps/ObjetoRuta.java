package carrito.gps;

import java.awt.*;

import carrito.media.*;
import java.io.Serializable;

public class ObjetoRuta implements Serializable {
  private double x, y, z, angulo, speed;

  private Dimension dim1, dim2;

  private int[] img1, img2;

  public ObjetoRuta(Media media, String disp1, String disp2,
                    double x, double y, double z, double angulo, double speed) {
    dim1 = new Dimension(media.getAncho(disp1), media.getAlto(disp1));
    dim2 = new Dimension(media.getAncho(disp2), media.getAlto(disp2));

    img1 = media.getImagen(disp1);
    img2 = media.getImagen(disp2);

    this.x = x;
    this.y = y;
    this.z = z;

    this.angulo = angulo;

    this.speed = speed;
  }

  public double getAngulo() {
    return angulo;
  }

  public Dimension getDim1() {
    return dim1;
  }

  public Dimension getDim2() {
    return dim2;
  }

  public int[] getImg1() {
    return img1;
  }

  public int[] getImg2() {
    return img2;
  }

  public double getSpeed() {
    return speed;
  }

  public double getX() {
    return x;
  }

  public double getY() {
    return y;
  }

  public double getZ() {
    return z;
  }

  public void setAngulo(double angulo) {
    this.angulo = angulo;
  }

  public void setDim1(Dimension dim1) {
    this.dim1 = dim1;
  }

  public void setDim2(Dimension dim2) {
    this.dim2 = dim2;
  }

  public void setImg1(int[] img1) {
    this.img1 = img1;
  }

  public void setImg2(int[] img2) {
    this.img2 = img2;
  }

  public void setSpeed(double speed) {
    this.speed = speed;
  }

  public void setX(double x) {
    this.x = x;
  }

  public void setY(double y) {
    this.y = y;
  }

  public void setZ(double z) {
    this.z = z;
  }
}
