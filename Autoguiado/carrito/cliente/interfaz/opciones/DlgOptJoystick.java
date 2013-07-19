/**
 * Contiene las clases correspondiente a la interfaz de usuario que permite
 * configurar las opciones
 */
package carrito.cliente.interfaz.opciones;

import java.awt.*;
import java.awt.event.*;
import java.io.*;

import javax.swing.*;

import carrito.configura.*;
import com.centralnexus.input.*;

/**
 * Clase que crea un asistente que va a permitir al usuario calibrar correctamente
 * un joystick con al menos dos ejes X e Y
 *
 * @author Néstor Morales Hernández
 * @version 1.0
 */

public class DlgOptJoystick extends JDialog implements Runnable, MouseListener, JoystickListener {
    // Identificadores de las etapas del Asistente
    private static final int INICIO = 0;
    private static final int CHECK_LEFT1 = 1;
    private static final int CHECK_LEFT2 = 2;
    private static final int CHECK_RIGHT1 = 3;
    private static final int CHECK_RIGHT2 = 4;
    private static final int CHECK_UP1 = 7;
    private static final int CHECK_UP2 = 8;
    private static final int CHECK_DOWN1 = 9;
    private static final int CHECK_DOWN2 = 10;
    private static final int CHECK_NONE1 = 5;
    private static final int CHECK_NONE2 = 6;
    private static final int FINAL = 11;

    /** Hilo de ejecución */
    private Thread hilo = null;

    /** Variable que contiene el estado actual en el asistente */
    private int estados = INICIO;

    // Variables de la interfaz
    private JPanel fondo = null;
    private JPanel botones = null;
    private JButton btnNext = null;
    private JButton btnLast = null;
    private JButton btnCancel = null;
    private JPanel principal = null;
    private JProgressBar pb = null;

    /**  Objeto que permite conocer el estado del Joystick */
    private static Joystick joy = null;

    // Variables que identifican las posiciones "extremo" del joystick
    private static float maxDerecha = -1.0f, maxIzquierda = 1.0f;
    private static float XNone = 0.0f, XDif = 0.0f;
    private static float YNone = 0.0f, YDif = 0.0f;
    private static float maxArriba = -1.0f, maxAbajo = 1.0f;
    private static float ejeX, ejeY, arriba, abajo;

    /** Variable que contiene los valores globales con los cuales va a trabajar
     * la aplicación
     * */
    private Constantes cte = null;

    /**
     * Contructor para la clase en el caso en el que el "padre" de la aplicación
     * es un JFrame. Llama al método {@link #init()} para que inicialice la interfaz
     * @param owner JFrame
     * @param cte Constantes
     */
    public DlgOptJoystick(JFrame owner, Constantes cte) {
        super(owner, true);
        this.cte = cte;
        init();
    }

    /**
     * Contructor para la clase en el caso en el que el "padre" de la aplicación
     * es un JDialog. Llama al método {@link #init()} para que inicialice la interfaz
     * @param owner JFrame
     * @param cte Constantes
     */
    public DlgOptJoystick(JDialog owner, Constantes cte) {
        super(owner, true);
        this.cte = cte;
        init();
    }

    /**
     * Inicializa la ventana del asistente y crea el objeto de monitorización del
     * Joystick. Llama a los métodos {@link #initImagen()} y {@link #reinit()} para
     * crear la imagen lateral y la interfaz de la parte principal de la ventana,
     * respectivamente
     */
    private void init() {
        getContentPane().setLayout(new BorderLayout());

        setTitle("Asistente para el calibrado del Joystick");
        Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
        setBounds((d.width - 640) / 2,(d.height - 516) / 2,640,516);
        initImagen();
        reinit();
        getContentPane().add(BorderLayout.CENTER, fondo);
        btnNext.addMouseListener(this);
        btnLast.addMouseListener(this);
        btnCancel.addMouseListener(this);

        // Obtiene el número de Joystick conectado a la computadora
        int numDevices = Joystick.getNumDevices();

        // Recorre todos los joystick y se queda con el primero que esté activo
        if (numDevices > 0) {
            try {
                for (int i = 0; i < numDevices; i++) {
                    if (Joystick.isPluggedIn(i)) {
                        joy = Joystick.createInstance(i);
                        joy.addJoystickListener(this);
                        break;
                    }
                }
            } catch(IOException ioe) {
                System.out.println("Error: " + ioe.getMessage());
            }
        }
    }

    /**
     * Inicializa el panel lateral y los botones
     */
    private void initImagen() {
        JPanel img = new JPanel();
        img.add(new JLabel(new ImageIcon("joyOpt.gif")));
        img.setBorder(BorderFactory.createLoweredBevelBorder());
        getContentPane().add(BorderLayout.WEST, img);

        fondo = new JPanel();
        principal = new JPanel();
        principal.setPreferredSize(new Dimension(530, 410));
        setResizable(false);
        btnNext = new JButton("Siguiente >>");
        btnLast = new JButton("<< Anterior");
        btnLast.setEnabled(false);
        btnCancel = new JButton("Cancelar");
        botones = new JPanel();
        botones.setLayout(new FlowLayout());
        botones.add(btnLast);
        botones.add(btnNext);
        botones.add(btnCancel);
        fondo.add(BorderLayout.CENTER, principal);
        fondo.add(BorderLayout.SOUTH, botones);
    }

    /**
     * Método que es llamado cada vez que hay un cambio de ventana en el asistente
     * y que la redibuja según la etapa en la que nos encontremos
     */
    private void reinit() {
        // Limpia la ventana
        principal.removeAll();
        principal.setLayout(new FlowLayout());
        principal.setPreferredSize(new Dimension(515, 445));
        // Decide en qué estado nos encontramos
        switch(estados) {
        // Estado inicial
        case INICIO:
            btnLast.setEnabled(false);
            JLabel lblInicio = new JLabel("<html>A continuación se van a llevar a cabo las " +
                                          "operaciones necesarias para efectuar el calibrado " +
                                          "del Joystick. Siga las intrucciones que se le vayan indicando</html>");
            lblInicio.setPreferredSize(new Dimension(510, 440));
            principal.add(lblInicio);
            break;
        // Instrucciones para comprobar el límite izquierdo
        case CHECK_LEFT1:
            JLabel lblCLeft = new JLabel("<html>Se va a realizar el calibrado del volante. " +
                                         "Gire todo lo que pueda el volante a la izquierda durante " +
                                         "3 segundos...</html>");
            lblCLeft.setPreferredSize(new Dimension(510, 440));
            principal.add(lblCLeft);
            break;
        // Ventana de comprobación del límite izquierdo
        case CHECK_LEFT2:
            JLabel lblCLeft2 = new JLabel("<html>Gire todo lo que pueda el volante a la izquierda durante los próximos 3 segundos...</html>");
            lblCLeft2.setPreferredSize(new Dimension(510, 220));
            pb = new JProgressBar(0, 3000);
            pb.setPreferredSize(new Dimension(450, 10));
            principal.add(lblCLeft2);
            principal.add(pb);
            hilo = new Thread(this);
            hilo.start();
            break;
        // Instrucciones para comprobar el límite derecho
        case CHECK_RIGHT1:
            JLabel lblCRight = new JLabel("<html>Se va a realizar el calibrado del volante. " +
                                         "Gire todo lo que pueda el volante a la derecha durante " +
                                         "3 segundos...</html>");
            lblCRight.setPreferredSize(new Dimension(510, 440));
            principal.add(lblCRight);
            break;
        // Ventana de comprobación del límite derecho
        case CHECK_RIGHT2:
            JLabel lblCRight2 = new JLabel("<html>Gire todo lo que pueda el volante a la derecha durante los próximos 3 segundos...</html>");
            lblCRight2.setPreferredSize(new Dimension(510, 220));
            pb = new JProgressBar(0, 3000);
            pb.setPreferredSize(new Dimension(450, 10));
            principal.add(lblCRight2);
            principal.add(pb);
            hilo = new Thread(this);
            hilo.start();
            break;
        // Instrucciones para comprobar la posición central del volante
        case CHECK_NONE1:
            JLabel lblCNone = new JLabel("<html>Se va a realizar el calibrado del volante. " +
                                         "No toque el volante durante los próximos " +
                                         "3 segundos...</html>");
            lblCNone.setPreferredSize(new Dimension(510, 440));
            principal.add(lblCNone);
            break;
        // Ventana de comprobación de la posición central del volante
        case CHECK_NONE2:
            JLabel lblCNone2 = new JLabel("<html>No toque el volante ni el acelerador " +
                                          "durante los próximos 3 segundos...</html>");
            lblCNone2.setPreferredSize(new Dimension(510, 220));
            pb = new JProgressBar(0, 3000);
            pb.setPreferredSize(new Dimension(450, 10));
            principal.add(lblCNone2);
            principal.add(pb);
            hilo = new Thread(this);
            hilo.start();
            break;
        // Instrucciones para comprobar la aceleración máxima (eje Y)
        case CHECK_UP1:
            JLabel lblCUp = new JLabel("<html>Se va a realizar el calibrado del acelerador. " +
                                         "Apriete todo lo que pueda el acelerador durante " +
                                         "3 segundos...</html>");
            lblCUp.setPreferredSize(new Dimension(510, 440));
            principal.add(lblCUp);
            break;
        // Ventana de comprobación de la aceleración máxima (eje Y)
        case CHECK_UP2:
            JLabel lblCUp2 = new JLabel("<html>Apriete todo lo que pueda el acelerador durante los próximos 3 segundos...</html>");
            lblCUp2.setPreferredSize(new Dimension(510, 220));
            pb = new JProgressBar(0, 3000);
            pb.setPreferredSize(new Dimension(450, 10));
            principal.add(lblCUp2);
            principal.add(pb);
            hilo = new Thread(this);
            hilo.start();
            break;
        // Instrucciones para comprobar el frenado máximo
        case CHECK_DOWN1:
            JLabel lblCDown = new JLabel("<html>Se va a realizar el calibrado del freno. " +
                                         "Apriete todo lo que pueda el freno durante " +
                                         "3 segundos...</html>");
            lblCDown.setPreferredSize(new Dimension(510, 440));
            principal.add(lblCDown);
            break;
        // Ventana de comprobación del frenado máximo
        case CHECK_DOWN2:
            JLabel lblCDown2 = new JLabel("<html>Apriete todo lo que pueda el freno durante los próximos 3 segundos...</html>");
            lblCDown2.setPreferredSize(new Dimension(510, 220));
            pb = new JProgressBar(0, 3000);
            pb.setPreferredSize(new Dimension(450, 10));
            principal.add(lblCDown2);
            principal.add(pb);
            hilo = new Thread(this);
            hilo.start();
            break;
        // Ventana de resultados
        case FINAL:
            btnLast.setEnabled(false);
            JLabel lblFin = new JLabel("<html>El asistente ha finalizado. Pulse 'Finalizar' para guardar los cambios " +
                                       "o 'Cancelar' para ignorarlos</html>");
            lblFin.setPreferredSize(new Dimension(510, 440));
            principal.add(lblFin);
            break;
        }
        principal.revalidate();
    }

    /**
     * Evento <i>mouseClicked</i>.
     * <p>Si se pulsa Cancelar, cierra la ventana e ignora los cambios</p>
     * <p>Si se pulsa Siguiente, va a la ventana siguiente</p>
     * <p>Si se pulsa Anterior, va a la ventana anterior</p>
     * <p>Si se pulsa Finalizar, guarda los cambios y cierra la ventana</p>
     * @param e MouseEvent
     */
    public void mouseClicked(MouseEvent e) {
        // Se debe haber pulsado el botón izquierdo
        if (e.getButton() != MouseEvent.BUTTON1)
            return;
        // Se pulsó cancelar
        if (e.getSource() == btnCancel) {
            this.setVisible(false);
            return;
        }
        // Comprueba el estado destino para cada uno de los estados según el
        // botón pulsado
        switch(estados) {
        case INICIO:
            if (e.getSource() == btnNext) {
                btnLast.setEnabled(true);
                estados = CHECK_LEFT1;
            }
            break;
        case CHECK_LEFT1:
            if (e.getSource() == btnNext) {
                estados = CHECK_LEFT2;
            } else if (e.getSource() == btnLast) {
                estados = INICIO;
            }
            break;
        case CHECK_LEFT2:
            if (e.getSource() == btnNext) {
                estados = CHECK_RIGHT1;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_LEFT1;
            }
            break;
        case CHECK_RIGHT1:
            if (e.getSource() == btnNext) {
                estados = CHECK_RIGHT2;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_LEFT1;
            }
            break;
        case CHECK_RIGHT2:
            if (e.getSource() == btnNext) {
                estados = CHECK_UP1;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_RIGHT1;
            }
            break;
        case CHECK_UP1:
            if (e.getSource() == btnNext) {
                estados = CHECK_UP2;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_RIGHT1;
            }
            break;
        case CHECK_UP2:
            if (e.getSource() == btnNext) {
                estados = CHECK_DOWN1;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_UP1;
            }
            break;
        case CHECK_DOWN1:
            if (e.getSource() == btnNext) {
                estados = CHECK_DOWN2;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_UP1;
            }
            break;
        case CHECK_DOWN2:
            if (e.getSource() == btnNext) {
                estados = CHECK_NONE1;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_DOWN1;
            }
            break;
        case CHECK_NONE1:
            if (e.getSource() == btnNext) {
                estados = CHECK_NONE2;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_DOWN1;
            }
            break;
        case CHECK_NONE2:
            if (e.getSource() == btnNext) {
                btnNext.setText("Finalizar");
                estados = FINAL;
            } else if (e.getSource() == btnLast) {
                estados = CHECK_NONE1;
            }
            break;
        case FINAL:
            if (e.getSource() == btnNext) {
                // NOTA: Al acelerar se obtienen valores negativos. Por eso
                // maxArriba debe ser menor que 0
                if (maxDerecha < 0 || maxIzquierda > 0 || maxArriba > 0 || maxAbajo < 0) {
                    Constantes.mensaje("Error al calibrar el volante. Vuelva a iniciar el asistente");
                } else {
                    // Establece los valores en la clase Constantes y los guarda en un fichero
                    cte.setJoystick(maxDerecha, maxIzquierda, maxArriba, maxAbajo, XNone, XDif, YNone, YDif);
                    cte.saveCliente();
                }
                this.setVisible(false);
                return;
            }
            break;
        }
        reinit();
    }
    /**
     * Evento <i>mouseEntered</i>
     * @param e MouseEvent
     */
    public void mouseEntered(MouseEvent e) {}
    /**
     * Evento <i>mouseExited</i>
     * @param e MouseEvent
     */
    public void mouseExited(MouseEvent e) {}
    /**
     * Evento <i>mousePressed</i>
     * @param e MouseEvent
     */
    public void mousePressed(MouseEvent e) {}
    /**
     * Evento <i>mouseReleased</i>
     * @param e MouseEvent
     */
    public void mouseReleased(MouseEvent e) {}

    /**
     * Evento <i>joystickAxisChanged</i>. Detecta un cambio en los ejes del
     * Joystick y actualiza el valor en la clase
     * @param j objeto que monitoriza el joystick y que debe coincidir con el
     * joystick que usa la clase
     */
    public void  joystickAxisChanged(Joystick j) {
        if (j != joy) return;
        ejeX = j.getX();
        ejeY = j.getY();
    }

    /**
     * Evento <i>joystickAxisChanged</i>. Detecta si se pulsó un botón
     * @param j objeto que monitoriza el oystick y que debe coincidir con el
     * joystick que usa la clase
     */
    public void joystickButtonChanged(Joystick j) {
    }

    /**
     * Método que lanza un hilo de ejecución, permitiendo de este modo actualizar
     * las variables que controlan cada uno de los ejes. Para cada estado, el hilo
     * de ejecución es diferente y dura 3 segundos (3000 milisegundos)
     */
    public void run() {

        switch(estados) {
        // Comprobación del límite izquierdo
        case CHECK_LEFT2: {
            long inicio = System.currentTimeMillis();
            int valor = 0;
            maxIzquierda = 1.0f;
            while ((valor = (int)(System.currentTimeMillis() - inicio)) < 3000) {
                if (maxIzquierda > ejeX)
                    maxIzquierda = ejeX;
                pb.setValue(valor);
            }
            pb.setValue(3000);
            break;
        }
        // Comprobación del límite derecho
        case CHECK_RIGHT2: {
            long inicio = System.currentTimeMillis();
            int valor = 0;
            maxDerecha = -1.0f;
            while ((valor = (int)(System.currentTimeMillis() - inicio)) < 3000) {
                if (maxDerecha < ejeX)
                    maxDerecha = ejeX;
                pb.setValue(valor);
            }
            pb.setValue(3000);
            break;
        }
        // Comprobación del centro
        case CHECK_NONE2: {
            long inicio = System.currentTimeMillis();
            int valor = 0;
            float sumaX = 0.0f, maxX = -10.0f, minX = 10.0f;
            float sumaY = 0.0f, maxY = -10.0f, minY = 10.0f;
            long size = 0;
            while ((valor = (int)(System.currentTimeMillis() - inicio)) < 3000) {
                pb.setValue(valor);
                sumaX += ejeX;
                sumaY += ejeY;
                maxX = (maxX < ejeX)? ejeX:maxX;
                minX = (minX > ejeX)? ejeX:minX;
                maxY = (maxY < ejeY)? ejeY:maxY;
                minY = (minY > ejeY)? ejeY:minY;
                size ++;
            }

            // Realiza una media de todos los valores recibidos
            XNone = sumaX / size;
            YNone = sumaY / size;

            float dif1, dif2;

            // Obtiene el rango central en el cual se considera 0 para el ejeX
            // Este va a ser la mayor de las distancias al eje tanto a la izquierda
            // como a la derecha (Se comprueba cuál de las dos es mayor y se toma en
            // ambos sentidos)
            dif1 = maxX - XNone;
            dif2 = XNone - minX;
            XDif = (dif1 > dif2)? dif1:dif2;

            // Lo mismo para el eje Y
            dif1 = maxY - YNone;
            dif2 = YNone - minY;
            YDif = (dif1 > dif2)? dif1:dif2;
            pb.setValue(3000);
            break;
        }
        // Comprobación de la aceleración máxima (en valor negativo, ya que la
        // aceleración la toma el Joystick como negativa y el freno como negativo)
        case CHECK_UP2: {
            long inicio = System.currentTimeMillis();
            int valor = 0;
            maxArriba = 1.0f;
            while ((valor = (int)(System.currentTimeMillis() - inicio)) < 3000) {
                if (maxArriba > ejeY)
                    maxArriba = ejeY;
                pb.setValue(valor);
            }
            pb.setValue(3000);
            break;
        }
        // Comprobación del frenado máximo
        case CHECK_DOWN2: {
            long inicio = System.currentTimeMillis();
            int valor = 0;
            maxAbajo = -1.0f;
            while ((valor = (int)(System.currentTimeMillis() - inicio)) < 3000) {
                if (maxAbajo < ejeY)
                    maxAbajo = ejeY;
                pb.setValue(valor);
            }
            pb.setValue(3000);
            break;
        }

        }
    }
}
