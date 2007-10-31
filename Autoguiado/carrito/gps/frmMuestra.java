package carrito.gps;

import javax.swing.*;
import java.awt.FlowLayout;

public class frmMuestra extends JFrame {
  public frmMuestra(CanvasMuestra[] canvas) {
    this.setDefaultCloseOperation(EXIT_ON_CLOSE);
    this.setCanvas(canvas);
  }

  public void setCanvas(CanvasMuestra[] canvas) {
    this.getContentPane().setLayout(new FlowLayout());
    int ancho = 0;
    int alto = 0;

    for (int i = 0; i < canvas.length; i++) {
      if (canvas[i].getWidth() > ancho) {
        ancho = canvas[i].getWidth();
      }
      alto += canvas[i].getHeight();
    }

    ancho += 100;
    alto += 100;

    this.setSize(ancho, alto);

    for (int i = 0; i < canvas.length; i++) {
      this.getContentPane().add(canvas[i]);
    }
    repaint();
  }
}
