package carrito.gps;


import carrito.media.Media;

import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.File;
import java.awt.Canvas;

// NOTA: Se asume que el espacio de color va a ser RV24
public class CapturaImagen {
  static {
    System.loadLibrary("javavlc");
  }

  private String dispositivo = "";
  private int ancho = -1;
  private int alto = -1;
  private Media media = null;

  private native void _verImagen(String dispositivo);

  public CapturaImagen(String vlc, String dispositivo, boolean startCamara) {
    this.dispositivo = dispositivo;
    media = new Media(vlc);

    if (startCamara)
      startCamara();

    while((ancho = media.getAncho(dispositivo)) < 0) {
      try {
        Thread.sleep(100);
      } catch (Exception e) {}
    }
    alto = media.getAlto(dispositivo);
  }

  // Este método está disponible sólo para pruebas
  public void startCamara() {
    media.addInstancia("dshow:// :dshow-vdev=" + dispositivo.replaceAll(" ", "&nbsp;") +
                       " :dshow-adev=none :dshow-size=320x240");
  }

  public ImagenId getImagen(double x, double y, double angulo) {
    int imagen[] = media.getImagen(dispositivo);
    ImagenId ii = new ImagenId(x, y, angulo, ancho, alto, imagen);
    ii.setNombre(dispositivo);
    return ii;
  }

  public void verImagen() {
    _verImagen(dispositivo);
  }

  public void saveImagen(String fichero) {
    media.saveImagenActual(dispositivo, fichero);
  }

  public static void main(String args[]) {
  String dispositivo = "Creative WebCam Notebook Ultra #3:0";
  CapturaImagen ci = new CapturaImagen(args[0], dispositivo, false);
  int i = 0;
  while (true) {
    i++;
    ImagenId ii = ci.getImagen(i * 10 % 100, i * 20 % 100, i * 30 % 100);
    ii.setNombre(dispositivo);
    ii.verImagen();
    //ci.verImagen();
    /*try {
      Thread.sleep(1000);
    } catch (Exception e) {}*/
  }
  }
}
