/**
 * 
 */
package sibtra.gps;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.KeyStroke;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 * Permite la edición de ficheros de ruta
 * 
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class EditaFicherosRuta extends JFrame implements  ItemListener, ActionListener, ChangeListener, MouseListener {

	private Ruta rutaEspacial=null, rutaTemporal=null, rutaActual=null;
	private Trayectoria traEspacial=null, traTemporal=null, traActual=null;
	private PanelExaminaRuta per;
	
	/** Barra de menu ventana principal */
	JMenuBar barraMenu;
	/** Menu de archivo dentro de la ventana principal */
	JMenu menuArchivo;
	/** Boton de salir del menu de archivo */
	private JMenuItem miSalir;
	/** Boton de abrir ruta de archivo */
	private JMenuItem miAbrir;

	private JFileChooser fc;
	private JLabel jlNomF;
	private JCheckBox jcbTemporal;
	private PanelMuestraRuta pmr;
	private JCheckBox jcbMarcarDM;
	
	public EditaFicherosRuta(String titulo) {
		
		super(titulo);
		fc=new JFileChooser(new File("./Rutas"));
		JPanel cp=(JPanel)getContentPane();
		cp.setLayout(new BoxLayout(cp,BoxLayout.PAGE_AXIS));
		pmr=new PanelMuestraRuta(null);
		cp.add(pmr);
		pmr.getJPanelGrafico().addMouseListener(this); //para recibir el ratón
		per=new PanelExaminaRuta();
		cp.add(per);

        //barra de menu
        barraMenu=new JMenuBar();
        setJMenuBar(barraMenu); //ponemos barra en la ventana
        //menu de archivo
        menuArchivo=new JMenu("Fichero");
        barraMenu.add(menuArchivo);

        miAbrir=new JMenuItem("Abrir Ruta");
        miAbrir.setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_O,KeyEvent.CTRL_MASK));
        miAbrir.addActionListener(this);
        menuArchivo.add(miAbrir);

        miSalir=new JMenuItem("Salir");
        miSalir.setAccelerator(KeyStroke.getKeyStroke( KeyEvent.VK_Q,KeyEvent.CTRL_MASK));
        miSalir.addActionListener(this);
        menuArchivo.add(miSalir);
        //lo añadimos al menu al final

		{
			JPanel jpSur=new JPanel();
			jlNomF=new JLabel("Fichero: ");
			jpSur.add(jlNomF);

			jcbTemporal=new JCheckBox("Mostrar Temporal");
			jcbTemporal.addItemListener(this);
			jpSur.add(jcbTemporal);

			jcbMarcarDM=new JCheckBox("Marcar usados DM");
			jcbMarcarDM.addItemListener(this);
			jpSur.add(jcbMarcarDM);

			cp.add(jpSur);
		}
		
		per.jsDato.addChangeListener(this);
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		pack();
		setVisible(true);
		setBounds(0, 384, 1024, 742);
	}

	public Ruta getRutaTemporal(){
		return rutaTemporal;
	}

	public Ruta getRutaEspacial(){
		return rutaEspacial;
	}

	public void actionPerformed(ActionEvent e) {
		if(e.getSource()==miSalir) {
			Terminar();
			return;
		}
//		if(e.getSource()==jmiAcercaDe) {
//		JOptionPane.showMessageDialog(ventanaPrincipal, "VERDINO: ISAATC ULL");
//		}
		if(e.getSource()==miAbrir) {
			int devuelto=fc.showOpenDialog(this);
			if(devuelto==JFileChooser.APPROVE_OPTION) {
				File file=fc.getSelectedFile();
				try {
					ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
					rutaEspacial=(Ruta)ois.readObject();
					rutaTemporal=(Ruta)ois.readObject();
					ois.close();
					traEspacial=new Trayectoria(rutaEspacial);
					traTemporal=new Trayectoria(rutaTemporal);
					eligeRuta();
					jlNomF.setText("Fichero: "+file.getName());
				} catch (IOException ioe) {
					System.err.println("Error al abrir el fichero " + file.getName());
					System.err.println(ioe.getMessage());
				} catch (ClassNotFoundException cnfe) {
					System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
				}
			}
			return;
		}
	}
	
	/** Elige ruta activa según {@link #jcbTemporal} */
	private void eligeRuta() {
		if (jcbTemporal.isSelected()) {
			per.setRuta(rutaTemporal);
			pmr.setRuta(rutaTemporal);
			rutaActual=rutaTemporal;
		} else {
			per.setRuta(rutaEspacial);
			pmr.setRuta(rutaEspacial);
			rutaActual=rutaEspacial;
		}
		traActual=new Trayectoria(rutaActual);
	}

    /** Metodo para terminar la ejecución */
    protected void Terminar() {
    	System.out.println("Terminamos ...");
		System.exit(0);
	}

	public void itemStateChanged(ItemEvent e) {
		if(e.getSource()==jcbTemporal)
			eligeRuta();
		if(e.getSource()==jcbMarcarDM);
		
			if(e.getStateChange()==ItemEvent.SELECTED) {
				pmr.setMarcados(per.ruta.indiceConsideradosDM);
				pmr.actualiza();
			} else { 
				pmr.setMarcados(null);
				pmr.actualiza();
			}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new EditaFicherosRuta("Examina Fich Ruta");
	}


	public void stateChanged(ChangeEvent arg0) {
		GPSData npto=per.ruta.getPunto((Integer)per.jsDato.getValue());
		pmr.nuevoPunto(npto);
	}
	
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}
	
	/** 
	 * Si es pulsación 1 seleccionamos punto más cercano para representar
	 * Si es 3 mostramos menu del punto
	 * @param e
	 */
	public void mousePressed(MouseEvent even) {
		if(rutaActual==null)
			return; //no hay ruta
		//Buscamos índice del punto más cercano de la ruta correspondiente
		Point2D.Double pto=pmr.pixel2Point(even.getX(), even.getY());
		System.out.println(getClass().getName()+": Pulsado Boton "+even.getButton()
				+" en posición: ("+even.getX()+","+even.getY()+")"
				+"  ("+pto.getX()+","+pto.getY()+")  "
		);
		//indice y distancia del más cercano usando la trayectoria
		traActual.situaCoche(pto.getX(),pto.getY());
		double distMin=traActual.distanciaAlMasCercano();
		int indMin=traActual.indiceMasCercano();
		double escala=pmr.getEscala();
		System.out.println(getClass().getName()+": Punto más cercano a "+distMin
				+" indice:"+indMin+ " escala:"+escala+ " veintaba:"+escala/10);
		if(!Double.isNaN(escala)) {
			if(distMin>(escala/10))
				return; //pulsación muy lejana
		} else if (distMin>2)
			return; //si fuera de escala la distancia debe ser <2

		if(even.getButton()==MouseEvent.BUTTON1)  {
			//fijamos el indice correspondiente
			per.jsDato.setValue(indMin);			
		}
		if(even.getButton()==MouseEvent.BUTTON3)  {
			System.out.println("Mostramos el menu de punto");
			mostrarMenu(indMin, even);			
		}


	}
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stubTODO Auto-generated method stubTODO Auto-generated method stub
		
	}

	public void mouseEntered(MouseEvent e) {
		// No hacemos nada por ahora
		
	}
	public void mouseExited(MouseEvent e) {
		//  No hacemos nada por ahora
		
	}

	private void mostrarMenu(final int ipto, MouseEvent me) {
		JPopupMenu popup = new JPopupMenu();
		JMenuItem item = new JMenuItem("Borrar Punto");
		popup.add(item);
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				rutaActual.remove(ipto);
				eligeRuta();
			}
		});
		popup.show(me.getComponent(), me.getX(), me.getY());

	}

}
