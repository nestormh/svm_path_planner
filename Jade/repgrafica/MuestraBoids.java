package repgrafica;

import sibtra.gps.Trayectoria;
import sibtra.lms.BarridoAngular;
import sibtra.lms.ManejaLMS;
import sibtra.lms.PanelMuestraBarrido;
import sibtra.lms.ManejaLMS111;
import sibtra.log.VentanaLoggers;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
//import java.awt.geom.Line2D.Double;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sun.reflect.ReflectionFactory.GetReflectionFactoryAction;

import com.bruceeckel.swing.Console;

import Jama.Matrix;
import boids.*;

class Dibujante2 extends JPanel{
	double largoCoche = 1.7;
	double anchoCoche = 1;
	Matrix posCoche = new Matrix(2,1);
	double yawCoche;
	Vector<Boid> bandadaPintar;// = new Vector<Boid>();
	Vector<Obstaculo> obstaculosPintar;// = new Vector<Obstaculo>();
	Vector<Matrix> rutaDinamica = null;
	
	public Dibujante2(){
		bandadaPintar = new Vector<Boid>();
		obstaculosPintar = new Vector<Obstaculo>();
	}
	
	public double getYawCoche() {
		return yawCoche;
	}

	public void setYawCoche(double yawCoche) {
		this.yawCoche = yawCoche;
	}


	public Matrix getPosCoche() {
		return posCoche;
	}

	public void setPosCoche(Matrix posIni) {
		this.posCoche = posIni;
	}
	
	public void introducirBoid(Boid b){
		bandadaPintar.add(b);		
	}
	
	public void introducirBandada(Vector<Boid> banda){
		bandadaPintar = banda;
//		bandadaPintar.clear();
//		for (int i =0;i<banda.size();i++){
//			bandadaPintar.add(banda.elementAt(i));
//		}
	}
	
	public void introducirObstaculo(Obstaculo b){
		obstaculosPintar.add(b);		
	}
	
	public void introducirObstaculos(Vector<Obstaculo> banda){
		for (int i =0;i<banda.size();i++){
			obstaculosPintar.add(banda.elementAt(i));
		}
	}
	
	public void eliminarObstáculos(){
		obstaculosPintar.clear();
	}
	
	public Vector<Boid> getBandadaPintar(){
		return bandadaPintar;
	}
	public Vector<Obstaculo> getObstaculoPintar(){
		return obstaculosPintar;
	}
	
	public void setRutaDinamica(Vector<Matrix> ruta){
		this.rutaDinamica = ruta;
	}
	/**
	 * Coordenadas de la esquina superior izquierda.
	 * En unidades mundo real.
	 */
	public Point2D esqSuperiorIzquierda;
	
		/**
	 * Coordenadas de la esquina inferior derecha.
	 * En unidades mundo real.
	 */
	public Point2D esqInferiorDerecha;
	
	public Point2D getEsqSuperiorIzquierda() {
		return esqSuperiorIzquierda;
	}

	public void setEsqSuperiorIzquierda(Point2D esqSuperiorIzquierda) {
		this.esqSuperiorIzquierda = esqSuperiorIzquierda;
	}
	
	public void setEsqSuperiorIzquierda(double x, double y) {
		this.esqSuperiorIzquierda = new Point2D.Double(x,y);
	}

	public Point2D getEsqInferiorDerecha() {
		return esqInferiorDerecha;
	}

	public void setEsqInferiorDerecha(Point2D esqInferiorDerecha) {
		this.esqInferiorDerecha = esqInferiorDerecha;
	}

	public void setEsqInferiorDerecha(double x, double y) {
		this.esqInferiorDerecha = new Point2D.Double(x,y);
	}
	
	/**
	 * Convierte punto en el mundo real a punto en la pantalla.
	 * @param x coordenada X del punto
	 * @param y coordenada Y del punto
	 * @return punto en pantalla
	 */
	protected Point2D.Double point2Pixel(double x, double y) {
		return new Point2D.Double(x*this.getWidth()/esqInferiorDerecha.getX(),
				this.getHeight()-(y*(this.getHeight()/esqSuperiorIzquierda.getY())));
	}
	/**
	 * Convierte punto de la pantalla a coordenadas reales
	 * @param x medida en píxeles de x
	 * @param y medida en píxeles de y
	 * @return
	 */
	protected Point2D.Double pixel2Point(int x, int y){
		return new Point2D.Double(x*esqInferiorDerecha.getX()/this.getWidth(),
				esqSuperiorIzquierda.getY()-(y*(esqSuperiorIzquierda.getY()/this.getHeight())));
	}
	
//	private int incrNuevosBoids = 3;
//	private int contNuevosBoids = incrNuevosBoids ;
	private int contIteraciones = 0;
	
	public void paintComponent(Graphics g2) {
		Graphics2D g3 = (Graphics2D) g2;		
		super.paintComponent(g3);
		Matrix centroMasa = new Matrix(2,1);
		
		//------------------------- Pinto los Boids----------------------------
		
		if (bandadaPintar.size() > 0){		
			contIteraciones++;
			for (int i=0;i<bandadaPintar.size();i++){
		
				g3.setColor(Color.blue);

				Point2D pixel = point2Pixel(bandadaPintar.elementAt(i).getPosicion().get(0,0),
						bandadaPintar.elementAt(i).getPosicion().get(1,0));
				g3.drawOval((int)pixel.getX()-2,(int)pixel.getY()-2,4,4);
//				g3.drawOval((int)bandadaPintar.elementAt(i).getPosicion().get(0,0)-2,
//						(int)bandadaPintar.elementAt(i).getPosicion().get(1,0)-2,4,4);
				
//				g3.draw(bandadaPintar.elementAt(i).getLineaDireccion());
//				GeneralPath ruta = new GeneralPath();
//				Point2D pixelCoche = point2Pixel(posCoche.get(0, 0),posCoche.get(1, 0));
//				ruta.moveTo(pixelCoche.getX(),pixelCoche.getY());
//				for(int k=1;k<bandadaPintar.elementAt(i).getRutaBoid().size();k++){
//					pixel = point2Pixel(bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(0, 0),
//							bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(1, 0));
//				ruta.lineTo(pixel.getX(),pixel.getY());
//				}
//				g3.draw(ruta);
				centroMasa = centroMasa.plus(bandadaPintar.elementAt(i).getPosicion());
			}
		}
		
		/*------------------------Pinto el coche----------------------------------*/

		g3.setColor(Color.GRAY);

		Point2D pxDD= point2Pixel(posCoche.get(0,0)+anchoCoche/2*Math.sin(yawCoche),
				posCoche.get(1,0)-anchoCoche/2*Math.cos(yawCoche));
		Point2D pxDI= point2Pixel(posCoche.get(0,0)-anchoCoche/2*Math.sin(yawCoche),
				posCoche.get(1,0)+anchoCoche/2*Math.cos(yawCoche));
		Point2D pxPD=point2Pixel(posCoche.get(0,0)+anchoCoche/2*Math.sin(yawCoche)
				-largoCoche*Math.cos(yawCoche),
				posCoche.get(1,0)-anchoCoche/2*Math.cos(yawCoche)
				-largoCoche*Math.sin(yawCoche));
		Point2D pxPI= point2Pixel(posCoche.get(0,0)-anchoCoche/2*Math.sin(yawCoche)
				-largoCoche*Math.cos(yawCoche),
				posCoche.get(1,0)+anchoCoche/2*Math.cos(yawCoche)
				-largoCoche*Math.sin(yawCoche));
		GeneralPath coche=new GeneralPath();
		coche.moveTo((float)pxDD.getX(),(float)pxDD.getY());
		coche.lineTo((float)pxPD.getX(),(float)pxPD.getY());
		coche.lineTo((float)pxPI.getX(),(float)pxPI.getY());
		coche.lineTo((float)pxDI.getX(),(float)pxDI.getY());
		coche.closePath();		
		g3.draw(coche);
		g3.fill(coche);
		
//		//---------------------- Pinto el centro de masa--------------------------------
		
		centroMasa.timesEquals((double)1/(double)bandadaPintar.size());		
		Point2D centroMasaPixel = point2Pixel(centroMasa.get(0, 0),
				centroMasa.get(1, 0));
		g2.setColor(Color.cyan);
		g2.drawOval((int)centroMasaPixel.getX()-2,(int)centroMasaPixel.getY()-2,4,4);
		
		//------------------------ Pinto los obstáculos---------------------------------
		
		for (int i=0;i<obstaculosPintar.size();i++){
			g3.setColor(Color.red);
			Point2D obst = point2Pixel(obstaculosPintar.elementAt(i).getPosicion().get(0,0)-
					obstaculosPintar.elementAt(i).getLado()/2,
					obstaculosPintar.elementAt(i).getPosicion().get(1,0)+
					obstaculosPintar.elementAt(i).getLado()/2);
			double lado = obstaculosPintar.elementAt(i).getLado()*this.getWidth()/esqInferiorDerecha.getX();
			Rectangle2D cuadrado = new Rectangle2D.Double(obst.getX(),
					obst.getY(),lado,lado);
			g3.draw(cuadrado);
			g3.fill(cuadrado);
			obst = point2Pixel(obstaculosPintar.elementAt(i).getPosicion().get(0,0)-Boid.getRadioObstaculo(),
					obstaculosPintar.elementAt(i).getPosicion().get(1,0)+Boid.getRadioObstaculo());
			double radio = Boid.getRadioObstaculo()*this.getWidth()/esqInferiorDerecha.getX();
			g3.drawOval((int)obst.getX(),(int)obst.getY(),(int)radio*2,(int)radio*2);
//			obst = point2Pixel(obstaculosPintar.elementAt(i).getPosicion().get(0,0)-Boid.getRadioObstaculoLejos(),
//					obstaculosPintar.elementAt(i).getPosicion().get(1,0)+Boid.getRadioObstaculoLejos());
//			g3.setColor(Color.black);
//			radio = Boid.getRadioObstaculoLejos()*this.getWidth()/esqInferiorDerecha.getX();
//			g3.drawOval((int)obst.getX(),(int)obst.getY(),(int)radio*2,(int)radio*2);
			
		}
		// Pinto el objetivo
		g3.setColor(Color.magenta);
		Point2D objetivo = point2Pixel(Boid.getObjetivo().get(0,0),Boid.getObjetivo().get(1,0));
		g3.drawOval((int)objetivo.getX(),(int)objetivo.getY(),5,5);
		
		//----------------------- Pinto la ruta dinámica---------------------------------
		
		if (rutaDinamica != null){
			if (rutaDinamica.size() > 0){
				
				GeneralPath rutaDinamic = new GeneralPath();
				Point2D ptoRuta = point2Pixel(rutaDinamica.elementAt(0).get(0,0),
						rutaDinamica.elementAt(0).get(1,0));
				rutaDinamic.moveTo(ptoRuta.getX(),ptoRuta.getY());
				for(int k=1;k<rutaDinamica.size();k++){
					ptoRuta = point2Pixel(rutaDinamica.elementAt(k).get(0,0),
							rutaDinamica.elementAt(k).get(1,0));
					rutaDinamic.lineTo(ptoRuta.getX(),ptoRuta.getY());
				}
				g3.draw(rutaDinamic);
			}
		}
	}		
}


public class MuestraBoids extends JApplet implements ChangeListener,ActionListener,MouseListener{
	
	public Vector<Matrix> rutaDinamica;
	public Vector<Matrix> getRutaDinamica() {
		return rutaDinamica;
	}

	public void setRutaDinamica(Vector<Matrix> rutaDinamica) {
		this.rutaDinamica = rutaDinamica;
	}
	boolean batch = false;
	private boolean objetivoEncontrado;
	private boolean play = false;
	private boolean colocandoObs = false;
	private boolean colocandoBan = false;
	private boolean pintarEscena = false;
	VentanaLoggers ventLogs = new VentanaLoggers();
	Simulador sim;// = new Simulador();
	JFileChooser selectorArchivo = new JFileChooser(new File("./Escenarios"));
	JMenuBar barraMenu = new JMenuBar(); 
	JMenu menuArchivo = new JMenu("Archivo");
	JMenu menuBandada = new JMenu("Bandada");
	Dibujante2 pintor;// = new Dibujante2();
	JLabel etiquetaPesoLider = new JLabel("Liderazgo");
	SpinnerNumberModel spPesoLider = new SpinnerNumberModel(Boid.getPesoLider(),0,100,0.1);
	JSpinner spinnerPesoLider = new JSpinner(spPesoLider);
	JLabel etiquetaCohesion = new JLabel("Cohesión");
	SpinnerNumberModel spCohesion = new SpinnerNumberModel(Boid.getPesoCohesion(),0,100,0.1);
	JSpinner spinnerCohesion = new JSpinner(spCohesion);
	JLabel etiquetaSeparacion = new JLabel("Separación");
	SpinnerNumberModel spSeparacion = new SpinnerNumberModel(Boid.getPesoSeparacion(),0,1000,0.1);
	JSpinner spinnerSeparacion = new JSpinner(spSeparacion);
	JLabel etiquetaAlineacion = new JLabel("Alineación");
	SpinnerNumberModel spAlineacion = new SpinnerNumberModel(Boid.getPesoAlineacion(),0,100,0.1);
	JSpinner spinnerAlineacion = new JSpinner(spAlineacion);
	JLabel etiquetaObjetivo = new JLabel("Vel Obj");
	SpinnerNumberModel spObjetivo = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,100,0.1);
	JSpinner spinnerObjetivo = new JSpinner(spObjetivo);
	JLabel etiquetaEvitaObs = new JLabel("Evitar Obst");
	SpinnerNumberModel spEvitaObs = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,0.1);
	JSpinner spinnerEvitaObs = new JSpinner(spEvitaObs);
//	JLabel etiquetaObsCerca = new JLabel("Obst cerca");
//	SpinnerNumberModel spEvitaObsCerca = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,0.1);
//	JSpinner spinnerEvitaObsCerca = new JSpinner(spEvitaObsCerca);
	JLabel etiquetaCompLateral = new JLabel("Comp lateral");
	SpinnerNumberModel spCompLateral = new SpinnerNumberModel(Boid.getPesoCompensacionLateral(),0,1000,0.1);
	JSpinner spinnerCompLateral = new JSpinner(spCompLateral);
	JLabel etiquetaRadObs = new JLabel("Radio obst");
	SpinnerNumberModel spRadioObs = new SpinnerNumberModel(Boid.getRadioObstaculo(),0,1000,0.1);
	JSpinner spinnerRadioObs = new JSpinner(spRadioObs);
//	JLabel etiquetaRadObsLejos = new JLabel("Radio lejos");
//	SpinnerNumberModel spRadioObsLejos = new SpinnerNumberModel(Boid.getRadioObstaculoLejos(),0,1000,0.1);
//	JSpinner spinnerRadioObsLejos = new JSpinner(spRadioObsLejos);
	JLabel etiquetaVelMax = new JLabel("Vel Máx");
	SpinnerNumberModel spVelMax = new SpinnerNumberModel(Boid.getVelMax(),0,100,1);
	JSpinner spinnerVelMax = new JSpinner(spVelMax);
	JLabel etiquetaNumBoids = new JLabel("Número de Boids");
	SpinnerNumberModel spNumBoids = new SpinnerNumberModel(20,1,200,1);
	JSpinner spinnerNumBoids = new JSpinner(spNumBoids);
	JButton pausa = new JButton("Play");
	JButton colocarObs = new JButton("Colocar obstáculos");
	JButton colocarBan = new JButton("Colocar la bandada");
	JMenuItem botonSalvar = new JMenuItem("Salvar escenario");
	JMenuItem botonCargar = new JMenuItem("Cargar escenario");
	JMenuItem botonBorrarBandada = new JMenuItem("Borrar Bandada");
	JMenuItem botonCrearBandada = new JMenuItem("Crear Bandada");
	JCheckBox checkBoxPintar = new JCheckBox("Dibujar");
	JButton configurar = new JButton("Configurar simulación batch");
	JButton simulacionBatch = new JButton("Ejecutar simulación batch");
	JLabel tiempoConsumido = new JLabel("0");
	public double tiempo;
	ConfigParam configurador;
	
	public void init(){
		Container cp = getContentPane();
		//Añadimos elemtentos al menu de bandada
		menuBandada.add(botonCrearBandada);
		menuBandada.add(botonBorrarBandada);
		menuBandada.add(colocarBan);
		barraMenu.add(menuBandada);
		//Añadimos elementos al menu de archivo
		menuArchivo.add(botonCargar);		
		menuArchivo.add(botonSalvar);		
		barraMenu.add(menuArchivo);
//		Añadimos elemtentos al menu de bandada
		menuBandada.add(botonCrearBandada);
		menuBandada.add(botonBorrarBandada);
		barraMenu.add(menuBandada);
		pintor = new Dibujante2();
		sim = new Simulador();
//		sim.getCp().iniciaNavega();
		pintor.introducirBandada(getSim().getBandada());
		pintor.addMouseListener(this);
		cp.add(pintor);
		this.setJMenuBar(barraMenu);
		JPanel panelSur = new JPanel(new FlowLayout());
		JPanel panelNorte = new JPanel(new FlowLayout());
//		panelSur.add(etiquetaPesoLider);
//		panelSur.add(spinnerPesoLider);
//		panelSur.add(etiquetaCohesion);
//		panelSur.add(spinnerCohesion);
		panelSur.add(etiquetaSeparacion);
		panelSur.add(spinnerSeparacion);
//		panelSur.add(etiquetaAlineacion);
//		panelSur.add(spinnerAlineacion);
		panelSur.add(etiquetaObjetivo);
		panelSur.add(spinnerObjetivo);
		panelSur.add(etiquetaEvitaObs);
		panelSur.add(spinnerEvitaObs);
//		panelSur.add(etiquetaObsCerca);
//		panelSur.add(spinnerEvitaObsCerca);
		panelSur.add(etiquetaCompLateral);
		panelSur.add(spinnerCompLateral);
		panelSur.add(etiquetaRadObs);
		panelSur.add(spinnerRadioObs);
//		panelSur.add(etiquetaRadObsLejos);
//		panelSur.add(spinnerRadioObsLejos);
		panelSur.add(etiquetaVelMax);
		panelSur.add(spinnerVelMax);
		panelNorte.add(checkBoxPintar);
		panelNorte.add(etiquetaNumBoids);
		panelNorte.add(spinnerNumBoids);
//		panelNorte.add(botonCrearBandada);
//		panelNorte.add(botonBorrarBandada);
		panelNorte.add(pausa);
		panelNorte.add(colocarObs);
		panelNorte.add(colocarBan);
		panelNorte.add(configurar);
		panelNorte.add(simulacionBatch);
		panelNorte.add(tiempoConsumido);
//		panelNorte.add(botonSalvar);
//		panelNorte.add(botonCargar);		
		spinnerPesoLider.addChangeListener(this);
		spinnerCohesion.addChangeListener(this);
		spinnerSeparacion.addChangeListener(this);
		spinnerAlineacion.addChangeListener(this);
		spinnerObjetivo.addChangeListener(this);
		spinnerEvitaObs.addChangeListener(this);
		spinnerRadioObs.addChangeListener(this);
		spinnerCompLateral.addChangeListener(this);
//		spinnerRadioObsCerca.addChangeListener(this);
//		spinnerRadioObsLejos.addChangeListener(this);
		spinnerVelMax.addChangeListener(this);
		spinnerNumBoids.addChangeListener(this);
		pausa.addActionListener(this);
		colocarObs.addActionListener(this);
		colocarBan.addActionListener(this);
		botonSalvar.addActionListener(this);
		botonCargar.addActionListener(this);
		botonBorrarBandada.addActionListener(this);
		botonCrearBandada.addActionListener(this);
		checkBoxPintar.addActionListener(this);
		configurar.addActionListener(this);
		simulacionBatch.addActionListener(this);
		cp.add(BorderLayout.SOUTH,panelSur);
		cp.add(BorderLayout.NORTH,panelNorte);
		// Creamos el simulador
//		sim = new Simulador();
	}
	
	public void salvarEscenario(String fichero,Simulador sim){
		try {
			File file = new File(fichero);
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
			oos.writeObject(sim.getObstaculos());
			oos.close();
		} catch (IOException ioe) {
			System.err.println("Error al escribir en el fichero " + fichero);
			System.err.println(ioe.getMessage());
		}
	}
	
	public void cargarEscenario(String fichero,Simulador sim) throws ClassNotFoundException{
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			sim.setObstaculos((Vector<Obstaculo>) ois.readObject());
			pintor.obstaculosPintar = sim.getObstaculos();
			repaint();
			ois.close();
		} catch (IOException ioe) {
			System.err.println("Error al leer en el fichero " + fichero);
			System.err.println(ioe.getMessage());
		}
	}
	
		

	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == spinnerPesoLider){
			Boid.setPesoLider(spPesoLider.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerCohesion){
			Boid.setPesoCohesion(spCohesion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerSeparacion){
			Boid.setPesoSeparacion(spSeparacion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerAlineacion){
			Boid.setPesoAlineacion(spAlineacion.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerObjetivo){
			Boid.setPesoObjetivo(spObjetivo.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerEvitaObs){
			Boid.setPesoObstaculo(spEvitaObs.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerEvitaObs){
			Boid.setPesoObstaculo(spEvitaObs.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerCompLateral){
			Boid.setPesoCompensacionLateral(spCompLateral.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerRadioObs){
			Boid.setRadioObstaculo(spRadioObs.getNumber().doubleValue());
		}
//		if (e.getSource() == spinnerRadioObsLejos){
//			Boid.setRadioObstaculoLejos(spRadioObsLejos.getNumber().doubleValue());
//		}
		if (e.getSource() == spinnerVelMax){
			Boid.setVelMax(spVelMax.getNumber().doubleValue());
		}
		if (e.getSource() == spinnerNumBoids){
			this.getSim().setTamanoBandada(spNumBoids.getNumber().intValue());
		}
	}

	public void actionPerformed(ActionEvent e){
		if (e.getSource() == pausa){
			if (!play){ // La etiqueta del botón cambia
				pausa.setText("Pausa");
				tiempo = System.currentTimeMillis();
				botonBorrarBandada.setEnabled(false);
				botonCrearBandada.setEnabled(false);
				setObjetivoEncontrado(false);
			}
			if (play){
				pausa.setText("Play");
				botonBorrarBandada.setEnabled(true);
				botonCrearBandada.setEnabled(true);
			}
			play = !play;
		}
		if (e.getSource() == colocarObs){
			if (!colocandoObs){ // La etiqueta del botón cambia
				colocarObs.setText("Colocando obstáculos");				
				colocarBan.setEnabled(false);
			}
			if (colocandoObs){
				colocarObs.setText("Colocar obstáculos");
				colocarBan.setEnabled(true);
			}
			colocandoObs = !colocandoObs;
		}
		if (e.getSource() == colocarBan){			
			if (!colocandoBan){ // La etiqueta del botón cambia
				colocarBan.setText("Colocando Bandada");
				colocarObs.setEnabled(false);
			}
			if (colocandoBan){
				colocarBan.setText("Colocar la Bandada");
				colocarObs.setEnabled(true);
			}
			colocandoBan = !colocandoBan;
		}
        if (e.getSource() == botonSalvar) {            
            int devuelto = selectorArchivo.showSaveDialog(null);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = selectorArchivo.getSelectedFile();
                salvarEscenario(file.getAbsolutePath(),this.getSim());
            }            
        }
        if (e.getSource() == botonCargar) {            
            int devuelto = selectorArchivo.showOpenDialog(null);
            if (devuelto == JFileChooser.APPROVE_OPTION) {
                File file = selectorArchivo.getSelectedFile();               
                try {
					cargarEscenario(file.getAbsolutePath(),this.getSim());
				} catch (ClassNotFoundException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
                
            }            
        }
        if (e.getSource() == botonBorrarBandada){
        	this.getSim().borrarBandada();
        	pintor.getBandadaPintar().clear();
        	pintor.repaint();
        }
        if (e.getSource() == botonCrearBandada){
//        	this.getSim().borrarBandsetTamanoBandadaada();
        	pintor.getBandadaPintar().clear();
        	this.getSim().crearBandada(spNumBoids.getNumber().intValue());
        }
        if (e.getSource() == checkBoxPintar){
        	pintarEscena = checkBoxPintar.isSelected();
        }
        if (e.getSource() == configurar){
        	if (configurador == null){
        		configurador = new ConfigParam();
        		Console.run(configurador,1000,300);
        	}        	        	
        }
        if (e.getSource() == simulacionBatch){
        	if(!batch){        		
        		if (getConfigurador().getVectorSim().size() > 0){
            		System.out.println("Existe un vector de puntos de diseño de tamaño "+getConfigurador().getVectorSim().size());
            		batch = true;
            		simulacionBatch.setText("Parar simulación batch");
            		colocarBan.setEnabled(false);
    				colocarObs.setEnabled(false);
    				pausa.setEnabled(false);
//            		simulacionBatch.setEnabled(false);
            	}
        	}
        	else{
        		simulacionBatch.setText("Ejecutar simulación batch");
        		colocarBan.setEnabled(true);
				colocarObs.setEnabled(true);
				pausa.setEnabled(true);
				batch = false;
        	}
        	        	
        }
	}
	

	public void mouseClicked(MouseEvent e) {
		if (colocandoObs){
			Point2D posicionReal = pintor.pixel2Point(e.getX(),e.getY());
			double pos[] = {posicionReal.getX(),posicionReal.getY()};
//			System.out.println("posición real del obstáculo "+posicionReal.getX()+" "+
//					posicionReal.getY());
			double vel[] = {0,0};
			double rumbo[] = {0,0};
			Matrix posicion = new Matrix(pos,2);			
			int i = 0;
			if(getSim().getObstaculos().size()%2 <= 0){
				vel[0] = 0;
				vel[1] = 2;
				rumbo[0] = 0;
				rumbo[1] = 2;
			}else{
				vel[0] = 0;
				vel[1] = -2;
				rumbo[0] = 0;
				rumbo[1] = -2;
			}
			Matrix velocidad = new Matrix(vel,2);
			Matrix rumboDes = new Matrix(rumbo,2);
			Obstaculo nuevoObs = new Obstaculo(posicion,velocidad,rumboDes);
			this.getSim().getObstaculos().add(nuevoObs);
			pintor.introducirObstaculo(nuevoObs);
//			System.out.println("Hay "+pintor.obstaculosPintar.size()+" obstáculos");
			repaint();
		}
		if (colocandoBan){
			Point2D posIniReal = pintor.pixel2Point(e.getX(),e.getY());
			double pos[] = {posIniReal.getX(),posIniReal.getY()};
			Matrix posicion = new Matrix(pos,2);
			this.getSim().posicionarBandada(posicion);
			this.getSim().setPosInicial(posicion);
			pintor.setPosCoche(posicion);
//			System.out.println(getSim().getBandada());
			pintor.introducirBandada(this.getSim().getBandada());
			repaint();
		}
		if (!colocandoBan && !colocandoObs){		
			Point2D objetivo = pintor.pixel2Point(e.getX(),e.getY());
			Boid.setObjetivo(objetivo.getX(),objetivo.getY());
			this.getSim().setObjetivo(objetivo.getX(),objetivo.getY());// cambia las coordenadas del objetivo
			repaint();
		}
		
	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated methodsetTamanoBandada stub
		
	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public double getTiempo() {
		return tiempo;
	}

	public void setTiempo(double tiempo) {
		this.tiempo = tiempo;
	}

	public boolean isObjetivoEncontrado() {
		return objetivoEncontrado;
	}

	public void setObjetivoEncontrado(boolean objetivoEncontrado) {
		this.objetivoEncontrado = objetivoEncontrado;
	}

	public Simulador getSim() {
		return sim;
	}

	public void setSim(Simulador sim) {
		this.sim = sim;
	}

	public ConfigParam getConfigurador() {
		return configurador;
	}

	public Dibujante2 getPintor() {
		return pintor;
	}

	public void setPintor(Dibujante2 pintor) {
		this.pintor = pintor;
	}
	public static void main(String[] args){
		
		//----------------------------------------//
		// Conectamos con el RF y lo configuramos //
		//----------------------------------------//
		
//		ManejaLMS m111=new ManejaLMS();
//		String resp;
//		BarridoAngular ba=null;
////		ba=m111.parseaBarrido("sRA LMDscandata 1 1 8D4C04 0 0 45FF 37 280F6 28A73 0 0 7 0 0 1388 168 0 1 DIST1 3F800000 00000000 FFF92230 1388 21D 7E 82 7D 79 83 76 7A 84 86 79 79 82 87 82 8A 90 7B 86 9B 88 7E 8C 97 A0 9F 9D 9D 9E AF A7 B2 A2 AE BA BC CA CA D2 E1 CD DF D1 E5 D3 E4 F4 FC 104 10C 108 11D 128 12B 147 143 15A 168 175 17B 184 18E 193 186 180 194 1A7 178 18B 197 19B 195 187 16B 177 179 172 170 17E 185 197 1A3 1F7 25E 27A 267 23E 1D3 1CA 1B6 1CA 1C1 1BF 1B8 1BA 1AD 191 19D 194 19F 192 17A 198 1BC 1F0 1E1 1E2 1CE 1CC 1BF 1C4 1C1 1FD 245 268 264 27B 276 283 273 283 29B 304 557 5AC 5BA 5D1 5CF 5DB 5EF 607 612 60D 612 620 629 64D 65D 659 66C 698 6A0 6AA 6C0 6D0 6CE 6DB 6F4 70C 70D 733 742 75A 76F 775 792 7A4 7B3 7E3 7E6 81E 82A 841 858 87C 8A8 8C0 8DD 907 941 953 943 923 923 8F6 8EB 8D3 8B9 8A1 8A8 86D 85C 84A 843 83A 81A 814 7F6 7F7 7D1 7D0 7C8 7AB 79D 799 792 777 770 767 752 74D 749 72F 72D 716 716 6FF 705 6F4 6FC 6DA 6E6 6CF 6CA 6B7 6AF 6AA 6A7 6AA 6A3 68E 687 6A8 67B 68B 66B 665 663 662 653 65B 64B 646 658 643 63A 634 62C 622 62B 628 62D 620 61A 62F 61A 617 612 608 60A 60C 60C 616 605 600 60B 61A 601 5FE 5F6 5FA 600 5FB 604 5FA 5F9 5F2 60B 5F2 5F5 5F0 5F5 5F0 5F8 5FB 5F1 5FC 5FC 5F3 605 5F8 5FF 5F9 5F9 600 5FB 610 5FE 60C 616 60B 609 610 610 60E 615 622 624 61F 61F 62B 632 628 62E 635 631 63F 642 64F 651 651 653 666 65E 65F 668 66F 677 68A 692 68C 688 698 6AA 6B5 6A7 6AE 6B4 6C0 6CB 6E5 6D9 6EB 6FB 6F5 701 704 711 71D 743 73A 753 774 78D 78B 79C 7A6 7A2 7B8 7D8 7D6 7E3 7FB 811 81C 826 841 82E 840 851 86C 86B 87A 88C 8A8 8B7 8CF 8E8 8F6 910 925 940 963 97D 995 9AF 9D4 9EF A01 A33 A3D A69 A8F AB6 AC7 AF9 B28 B4B B81 B94 BC8 BEC C23 C50 C7B C7E C9A CC8 D6B D51 D4B D38 D16 CF8 CF2 CE8 CD3 CC2 CAF CAF CA7 CB4 C9C C91 C7D C64 C69 C3F C3D C3C C38 C20 C11 C0F C06 BE4 BCB AE3 8D7 75F 768 76E 777 767 767 770 76E 768 761 764 757 75B 765 777 77B 773 765 8A1 9D9 B24 4C4 462 448 43D 45C 457 44E 41E 421 446 437 424 429 440 4D8 4A0 45E 427 454 450 460 467 472 462 46E 478 487 47A 48F 481 490 489 488 462 3E6 1B6 15A 13B F9 ED 102 EC D8 E2 DF E1 C8 D5 CF CD C8 C7 C3 BC BD A6 A2 9D 9B A6 A9 93 A9 9B 9A 9A 96 89 95 7A 90 8A 93 81 85 71 89 85 7E 75 72 64 6E 7A 74 75 71 7C 76 67 6D 67 0 0 1 7 Derecha 0 1 7B2 1 1 0 2A 27 28870 0");
//
//		System.out.println("Tratamos de conectar");
//		m111.conecta("192.168.0.3", 2111);
//
//		m111.enviaEspera("sMN SetAccessMode 03 F4724744");
//
//		//configuracion escaneo
//		m111.enviaEspera("sRN LMPscancfg");
//
//		//configuracion mensaje de datos escaneo
////		m111.enviaEspera("sWN LMDscandatacfg 01 00 0 1 00 00 00 00 00 +1");
//		m111.enviaEspera("sWN LMDscandatacfg 01 00 0 1 0 00 00 1 1 1 1 +1");
//
//		//7-segmento encendido
//		m111.enviaEspera("sMN mLMLSetDisp 07");
//
//		//Pedimos que comienze a medir
//		m111.enviaEspera("sMN LMCstartmeas");
//		
//		do {
//			//Miramos el status
//			try { Thread.sleep(1000); } catch (Exception e) {}
//			resp=m111.enviaEspera("sRN STlms");
//		} while(resp.charAt(10)!='7');
//
//		// Petición de envío de barrido por petición
//		
//		for(int i=1; i<5; i++) {
//			//Pedimos una medida
//			resp=m111.enviaEspera("sRN LMDscandata");
//			ba=m111.parseaBarrido(resp);
//			System.out.println("Barrido parseado:"+ba);
//			System.out.println("Distancia Maxima:"+ba.getDistanciaMaxima());
////			pmba.setBarrido(ba);
//			try { Thread.sleep(5000); } catch (Exception e) {}
//		}
//		
//		//iniciamos el modo contínuo
//		resp=m111.enviaEspera("sEN LMDscandata 1");
//			//esperamos una medida
//		resp=m111.leeMensaje();
//		ba=m111.parseaBarrido(resp);//			
//			
//		//Paramos el modo contínuo
////		resp=m111.enviaEspera("sEN LMDscandata 0");
//		
//		
//		
//		//paramos el LMS
////		m111.enviaEspera("sMN LMCstopmeas");
//		
//		// Método para desconectar el RF
////		m111.desconecta();
//		
//		//----------------------------------------//
//		// Fin de configuración y conexión del RF //
//		//----------------------------------------//
		
		MuestraBoids gui = new MuestraBoids();
//		Simulador simula = new Simulador();
//		gui.setSim(simula);
//		gui.getSim().setTamanoBandada(gui.getSim().getTamanoBandada());
//		gui.pintor.introducirBandada(gui.getSim().getBandada());
		Console.run(gui,1200,1000);
		int alturaPanel = gui.pintor.getHeight();
		int anchuraPanel = gui.pintor.getWidth();
		double longitudEscenario = 60;
		double anchuraEscenario = longitudEscenario*alturaPanel/anchuraPanel;
		gui.getSim().setAnchoEscenario(anchuraEscenario);
		gui.getSim().setLargoEscenario(longitudEscenario);
		gui.getPintor().setEsqInferiorDerecha(longitudEscenario,0);
		gui.getPintor().setEsqSuperiorIzquierda(0,anchuraEscenario);
		// generamos los obstáculos aleatoriamente
//		gui.getSim().generaObstaculos(8,1.5);
		int numSimu = 0;
		int simuDeseadas = 5;
		double distCercana = 2;
		Vector<Matrix> vectorPosCoche = new Vector<Matrix>();
		Vector<Double> yawCoche = new Vector<Double>();
		gui.pintor.introducirObstaculos(gui.getSim().getObstaculos());
		
		//creamos las zonas para clasificar las rutas
		
		// Indico la posición del objetivo y la posición inicial
		double[] objetivo = {gui.pintor.getEsqInferiorDerecha().getX(),
				gui.pintor.getEsqSuperiorIzquierda().getY()/2};
		gui.getSim().setObjetivo(new Matrix(objetivo,2));
		double[] inicial = {3,gui.pintor.getEsqSuperiorIzquierda().getY()/2};
		gui.getSim().setPosInicial(new Matrix(inicial,2));
		gui.pintor.setPosCoche(gui.getSim().getPosInicial());
		gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());
		
		//-------------------------------------------------------------------
		//----------------BUCLE PRINCIPAL------------------------------------
		//-------------------------------------------------------------------
		
		int indMinAnt = 0;
		double tMaximo = 100;
		double distCocheObjetivo = Double.POSITIVE_INFINITY;
		while (true){
//		while (numSimu < simuDeseadas){			
			if (gui.play){
				// reinicio todo para la siguiente simulación
				gui.getSim().setContIteraciones(0);
				vectorPosCoche.clear();
				yawCoche.clear();
				gui.pintor.eliminarObstáculos();
				gui.getSim().generaObstaculos(10,1.5);				
				gui.pintor.introducirObstaculos(gui.getSim().getObstaculos());
				gui.getSim().borrarBandada();
				gui.getSim().crearBandada(20,gui.getSim().getContIteraciones());
				gui.getSim().setPosInicial(new Matrix(inicial,2));
				gui.pintor.setPosCoche(gui.getSim().getPosInicial());
				gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());
				distCocheObjetivo = gui.getSim().getObjetivo().minus(gui.getSim().getPosInicial()).norm2();
				double tAnt = System.currentTimeMillis()/1000;
				double tSim = System.currentTimeMillis()/1000;
				//La simulación acabará cuando el coche llegue al objetivo o cuando haya
				//transcurrido más tiempo del estipulado como bueno para una sola simulación
				//Para evitar que se atasque toda la simulación por lotes
				while((distCocheObjetivo > distCercana)&&(tSim-tAnt < tMaximo)){					
//					System.out.println("La duración de cada iteración es "+ (System.currentTimeMillis() - tAnt));
//					tAnt = System.currentTimeMillis();
//					resp=m111.leeMensaje();
//					ba=m111.parseaBarrido(resp);//
//					gui.getSim().posicionarObstaculos(ba);
//					gui.pintor.eliminarObstáculos();
//					gui.pintor.introducirObstaculos(gui.getSim().getObstaculos());
					gui.getSim().moverObstaculos();
					indMinAnt = gui.getSim().moverBoids(indMinAnt);
					int indice = 0;
//					double distObj = Double.POSITIVE_INFINITY;
//					for(int h = 0; h<gui.getSim().getBandada().size();h++){
//						double distan = gui.getSim().getBandada().elementAt(h).getDistObjetivo();
//						if(distan < distObj){
//							indice = h;
//							distObj = distan;
//						}
//					}
					double distOrigen = Double.POSITIVE_INFINITY;
					for(int h = 0; h<gui.getSim().getBandada().size();h++){
						double distan = gui.getSim().getBandada().elementAt(h).getDistOrigen();
						if(distan < distOrigen){
							indice = h;
							distOrigen = distan;
						}
					}

//					boolean flagUnaVez = true;
					gui.setRutaDinamica(gui.getSim().calculaRutaDinamica(indice));
					gui.pintor.setRutaDinamica(gui.getRutaDinamica());
					gui.getSim().moverPtoInicial(tAnt, gui.getSim().getTs());
					//Rellenamos los datos del camino seguido por el coche
					double[] posCocheAux = {gui.getSim().getPosInicial().get(0,0),
							gui.getSim().getPosInicial().get(1,0)};
					vectorPosCoche.add(new Matrix(posCocheAux,2));
//					vectorPosCoche.lastElement().print(10,4);
					yawCoche.add(gui.getSim().getModCoche().getYaw());
					//Medimos la distancia a la que se encuentra el coche del objetivo
					distCocheObjetivo = gui.getSim().getObjetivo().minus(gui.getSim().getPosInicial()).norm2();
					gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());
					/*if (gui.getRutaDinamica().size()>1){ //Comprobar si hay ruta que seguir					
						Trayectoria tr = new Trayectoria(gui.getSim().traduceRuta(gui.getRutaDinamica()));
						Trayectoria trMasPuntos = new Trayectoria(tr,0.1);
						trMasPuntos.situaCoche(gui.getSim().getModCoche().getX(),gui.getSim().getModCoche().getY());
						if (flagUnaVez){
							gui.getSim().getCp().setRuta(trMasPuntos);
							flagUnaVez = false;
						}					
						gui.getSim().moverPtoInicial(System.currentTimeMillis(),System.currentTimeMillis()-tAnt);
					}								
					gui.pintor.setPosCoche(gui.getSim().getPosInicial());
					gui.pintor.setYawCoche(gui.getSim().getModCoche().getYaw());*/
//					gui.getSim().simuPorLotes();				
					if (gui.pintarEscena)
						gui.pintor.repaint();	
					//recogemos el tiempo que transcurre en cada simulación
					tSim = System.currentTimeMillis()/1000;
				}
				numSimu++;
//				Poner aqui el cálculo de las medias, varianzas,etc del estudio estadístico
				double distEntrePtos = 0;
				int numDatos = vectorPosCoche.size()-1;
				double[] velCoche = new double[numDatos];
				double sumaVel = 0;
				double sumaYaw = 0;
				double mediaVel = 0;
				double mediaYaw = 0;
				double varianzaVel = 0;
				double varianzaYaw = 0;
				double sumaVarianzaVel = 0;
				double sumaVarianzaYaw = 0;
				double desvTipicaVel = 0;
				double desvTipicaYaw = 0;				
				//cálculo de las medias
//				System.out.println("posInicial de los boids");
//				System.out.println("-----------------------------------");
//				gui.getSim().getPosInicial().print(10,4);
//				System.out.println("-----------------------------------");
//				for (int i=0;i<vectorPosCoche.size();i++){
//					vectorPosCoche.elementAt(i).print(10,4);
//					System.out.println("yaw "+yawCoche.elementAt(i));
//				}
				for (int h=0;h<numDatos;h++){
					distEntrePtos = vectorPosCoche.elementAt(h+1).minus(vectorPosCoche.elementAt(h)).norm2();
//					System.out.println("dist entre puntos "+ distEntrePtos);
					//distEntrePtos/gui.getSim().getTs() = dist/tiempo = velocidad
					sumaVel = sumaVel + distEntrePtos/gui.getSim().getTs();
					velCoche[h] = distEntrePtos/gui.getSim().getTs();
					sumaYaw = sumaYaw + yawCoche.elementAt(h);
				}				
				mediaVel = sumaVel/numDatos;
				mediaYaw = sumaYaw/numDatos;
				//calculo las varianzas
				for (int k = 0;k<numDatos;k++){
					sumaVarianzaVel = sumaVarianzaVel + Math.sqrt(Math.abs(velCoche[k]-mediaVel));
					sumaVarianzaYaw = sumaVarianzaYaw + Math.sqrt(Math.abs(yawCoche.elementAt(k)-mediaYaw));
				}
				System.out.println("numDatos="+numDatos+" varianzaVel="+sumaVarianzaVel+" varianzaYaw="+sumaVarianzaYaw);
				varianzaVel = sumaVarianzaVel/numDatos;
				varianzaYaw = sumaVarianzaYaw/numDatos;
				desvTipicaVel = Math.sqrt(varianzaVel);
				desvTipicaYaw = Math.sqrt(varianzaYaw);
				//rellenamos los valores en el loger estadístico
				gui.getSim().getLogEstadistica().add(mediaVel,desvTipicaVel,mediaYaw,desvTipicaYaw);
				System.out.println("Acabó una simulación y calculó la estadística, los valores son "
						+mediaVel+" "+desvTipicaVel+" "+mediaYaw+" "+desvTipicaYaw);
			}
			if (gui.batch){
				gui.getSim().getBoidsOk().clear();
//				gui.getSim().setNumBoidsOkDeseados(gui.getConfigurador().getNumBoidsOk());
				gui.getSim().setTiempoMax(gui.getConfigurador().getTMax());
				for(int i=0;i<gui.getConfigurador().getVectorSim().size() && gui.batch;i++){
	    				gui.getSim().configurador(gui.getConfigurador().getVectorSim().elementAt(i),
	    						gui.getConfigurador().getNomParam());
	    				gui.getSim().crearBandada();
	    				gui.getSim().posicionarBandada(gui.getSim().getPosInicial());	    				
	    				gui.getSim().simuPorLotes();
	    				gui.tiempoConsumido.setText("Tardó " + gui.getSim().getTiempoInvertido() + " sec");	    				
	    				if (gui.pintarEscena){
	    					for(int j=0;j<gui.getSim().getBoidsOk().size();j++){
/*OJO! aqui se mejora la ruta*/			/*gui.getSim().getBoidsOk().elementAt(j).setRutaBoid(
	    								gui.getSim().mejoraRuta(
	    								gui.getSim().getBoidsOk().elementAt(j).getRutaBoid()));
	    						gui.getSim().getBoidsOk().elementAt(j).setRutaBoid(
	    								gui.getSim().mejoraRuta(
	    								gui.getSim().getBoidsOk().elementAt(j).getRutaBoid())); */
	    						//El identificador del tipo de camino será el número de vertices que tiene el
	    						//camino después de haber mejorado la ruta
	    						int identificador = gui.getSim().getBoidsOk().elementAt(j).getRutaBoid().size();
	    						//Compruebo si el vector de caminos está vacio
	    						if(gui.getSim().getCaminos().size()<=0){
	    							//Creo un tipo nuevo de camino y añado un boid al nº de boids que han tomado
	    							//este camino
	    							gui.getSim().getCaminos().add(new TipoCamino(identificador,1,
	    									gui.getSim().getBoidsOk().elementAt(j).getNumIteraciones(),
	    									gui.getSim().getBoidsOk().elementAt(j).calculaLongRuta()));
	    						}else{
	    							//Bucle para buscar coincidencias entre los identificadores existentes
		    						//y el identificador de la ruta del boid actual
		    						boolean encontrado = false;
		    						for(int k=0;k<gui.getSim().getCaminos().size() && !encontrado;k++){
		    							if(gui.getSim().getCaminos().elementAt(k).getIdentificador() == identificador){
		    								encontrado = true;
		    								gui.getSim().getCaminos().elementAt(k).addFrecuencia();
		    								gui.getSim().getCaminos().elementAt(k).addCoste(
		    										gui.getSim().getBoidsOk().elementAt(j).getNumIteraciones());
		    								gui.getSim().getCaminos().elementAt(k).addLongitud(
		    										gui.getSim().getBoidsOk().elementAt(j).calculaLongRuta());
		    							}
		    						}
		    						if(!encontrado){
		    							gui.getSim().getCaminos().add(new TipoCamino(identificador,1,		    									
			    							gui.getSim().getBoidsOk().elementAt(j).getNumIteraciones(),
			    							gui.getSim().getBoidsOk().elementAt(j).calculaLongRuta()));
		    						}
	    						}	
	    					}
	    					//Imprimimos en consola el resultado
	    					for(int q=0;q<gui.getSim().getCaminos().size();q++){
	    						System.out.println("La frec del camino tipo "+
	    								(gui.getSim().getCaminos().elementAt(q).getIdentificador()-1) +
	    								" es "+ gui.getSim().getCaminos().elementAt(q).getFrecuencia() + 
	    								" ,tiene un coste de " + gui.getSim().getCaminos().elementAt(q).calculaCoste() + 
	    								" y una longitud de " + gui.getSim().getCaminos().elementAt(q).getLongitudMedia());
	    					}
	    					gui.pintor.introducirBandada(gui.getSim().getBoidsOk());
	    					gui.pintor.repaint();     	    				
	    				}	    				
//	    				gui.pintor.introducirBandada(gui.getSim().getBandada());
//	    			}        				
	    				System.out.println("Ya acabó el simuporlotes");
	    				System.out.println(gui.getSim().getNumBoidsOk());
	    		}
				gui.simulacionBatch.setText("Ejecutar simulación batch");
        		gui.colocarBan.setEnabled(true);
				gui.colocarObs.setEnabled(true);
				gui.pausa.setEnabled(true);
				gui.batch = false;
//				gui.simulacionBatch.setEnabled(true);				
			}			
		}
//		System.out.println("Acabó todas las simulaciones");
	}
}