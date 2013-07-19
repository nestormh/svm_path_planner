package carrito.media;

import java.awt.Graphics;
import java.awt.Toolkit;
import java.awt.image.MemoryImageSource;
import java.awt.Image;
import java.awt.Canvas;
import java.awt.Color;

public class CanvasCaptura extends Canvas {
  private int[] imagen = null;
  private int ancho = -1;
  private int alto = -1;
  private Media media = null;
  private String dispositivo = "";

  public CanvasCaptura(Media media, String dispositivo) {
    this.media = media;
    this.dispositivo = dispositivo;

    /*imagen = Toolkit.getDefaultToolkit().createImage(
      new MemoryImageSource(ancho, alto, media.getImagen(dispositivo), 0, ancho));*/
  }

  public void setImagen() {
    ancho = media.getAncho(dispositivo);
    alto = media.getAlto(dispositivo);
    imagen = media.getImagen(dispositivo);
  }

  public void paint(Graphics g) {
    for (int j = 0; j < alto; j++) {
      for (int i = 0; i < ancho * 3; i += 3) {
        /*System.out.println((i / 3) + ", " + j + "-->" + (j * ancho * 3 + i));
        if ((imagen[i * ancho * 3 + j] > 255) ||
            (imagen[i * ancho * 3 + j] < 0)) {
          System.out.println("Error en " + (i * ancho * 3 + j) + ": " + imagen[i * ancho * 3 + j]);
        }
        if ((imagen[i * ancho * 3 + j + 1] > 255) ||
            (imagen[i * ancho * 3 + j + 1] < 0)) {
          System.out.println("Error en " + (i * ancho * 3 + j + 1) + ": " + imagen[i * ancho * 3 + j + 1]);
        }
        if ((imagen[i * ancho * 3 + j + 2] > 255) ||
            (imagen[i * ancho * 3 + j + 2] < 0)) {
          System.out.println("Error en " + (i * ancho * 3 + j + 2) + ": " + imagen[i * ancho * 3 + j + 2]);
        }*/

        if (imagen != null) {
          g.setColor(new Color(imagen[j * ancho * 3 + i + 2],
                               imagen[j * ancho * 3 + i + 1],
                               imagen[j * ancho * 3 + i]));
        }
        //System.out.println(g.getColor());
        g.fillRect(i / 3, alto - j - 1, 1, 1);
      }
    }
  }
}
