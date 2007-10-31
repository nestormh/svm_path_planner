package carrito.gps;


import java.awt.*;
import javax.swing.*;

public class CanvasMuestra extends JFrame {
  ImagenId imagen = null;

  public CanvasMuestra(ImagenId imagen) {
    this.imagen = imagen;
    this.setSize(new Dimension(imagen.getAncho(), imagen.getAlto()));
    this.setBackground(Color.red);
    this.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
    this.setVisible(true);
  }

/*  public void paint(Graphics2D g) {
   BufferedImage bi = new BufferedImage(imagen.getAncho(), imagen.getAlto(), BufferedImage.TYPE_INT_RGB);
         bi.setRGB(0,0,imagen.getAncho(), imagen.getAlto(), imagen.getImagen(), 0, imagen.getAncho());
         g.drawImage(bi, 0, 0, Color.white, null);*/
  public void paint(Graphics g) {
    int ancho = imagen.getAncho();
    int alto = imagen.getAlto();
    for (int j = 0; j < alto; j++) {
      for (int i = 0; i < ancho * 3; i += 3) {
        if (imagen != null) {
          g.setColor(new Color(imagen.getImagen()[j * ancho * 3 + i + 2],
                               imagen.getImagen()[j * ancho * 3 + i + 1],
                               imagen.getImagen()[j * ancho * 3 + i]));
        }
        g.fillRect(i / 3, alto - j - 1, 1, 1);
      }
    }
  }

  public void setImagen(ImagenId imagen) {
    this.imagen = imagen;
  }
}
