package repgrafica;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
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

import com.bruceeckel.swing.Console;

import Jama.Matrix;
import boids.*;
import boids.Obstaculo;

class Dibujante2 extends JPanel{
	Vector<Boid> bandadaPintar;// = new Vector<Boid>();
	Vector<Obstaculo> obstaculosPintar;// = new Vector<Obstaculo>();
	public Dibujante2(){
		bandadaPintar = new Vector<Boid>();
		obstaculosPintar = new Vector<Obstaculo>();
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
	
	public Vector<Boid> getBandadaPintar(){
		return bandadaPintar;
	}
	public Vector<Obstaculo> getObstaculoPintar(){
		return obstaculosPintar;
	}
	
	public void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;		
		super.paintComponent(g2);
		Matrix centroMasa = new Matrix(2,1);
		// Pinto los Boids
		if (bandadaPintar.size() > 0){		
			for (int i=0;i<bandadaPintar.size();i++){
				if (bandadaPintar.elementAt(i).isLider()){
					g2.setColor(Color.green);
//					g2.drawOval((int)bandadaPintar.elementAt(i).getPosicion().get(0,0)-(int)Boid.getRadioCohesion(),
//							(int)bandadaPintar.elementAt(i).getPosicion().get(1,0)-(int)Boid.getRadioCohesion(),
//							(int)Boid.getRadioCohesion()*2,(int)Boid.getRadioCohesion()*2);
					GeneralPath rutaLider = new GeneralPath();
					rutaLider.moveTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(0,0),
							bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(1,0));
					for(int k=1;k<bandadaPintar.elementAt(i).getRutaBoid().size();k++){
						rutaLider.lineTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(0,0),
								bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(1,0));
					}
					g2.draw(rutaLider);
				}
				else
					g2.setColor(Color.blue);
				g2.draw(bandadaPintar.elementAt(i).getForma());
				g2.fill(bandadaPintar.elementAt(i).getForma());
				g2.draw(bandadaPintar.elementAt(i).getLineaDireccion());
//				GeneralPath ruta = new GeneralPath();
//				ruta.moveTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(0,0),
//				bandadaPintar.elementAt(i).getRutaBoid().elementAt(0).get(1,0));
//				for(int k=1;k<bandadaPintar.elementAt(i).getRutaBoid().size();k++){
//				ruta.lineTo(bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(0,0),
//				bandadaPintar.elementAt(i).getRutaBoid().elementAt(k).get(1,0));
//				}
//				g2.draw(ruta);
				centroMasa = centroMasa.plus(bandadaPintar.elementAt(i).getPosicion());
			}
		}
		// Pinto el centro de masa
		centroMasa.timesEquals((double)1/(double)bandadaPintar.size());		
		g2.setColor(Color.cyan);
		g2.drawOval((int)centroMasa.get(0,0)-2,(int)centroMasa.get(1,0)-2,4,4);
		
		// Pinto los obstáculos
		for (int i=0;i<obstaculosPintar.size();i++){
			g2.setColor(Color.red);
			g2.draw(obstaculosPintar.elementAt(i).getForma());
			g2.fill(obstaculosPintar.elementAt(i).getForma());	
			
		}
		// Pinto el objetivo
		g2.setColor(Color.magenta);
		g2.drawOval((int)Boid.getObjetivo().get(0,0),(int)Boid.getObjetivo().get(1,0),5,5);
	}	
}


public class MuestraBoids extends JApplet implements ChangeListener,ActionListener,MouseListener{
	
	boolean batch = false;
	private boolean objetivoEncontrado;
	private boolean play = false;
	private boolean colocandoObs = false;
	private boolean colocandoBan = false;
	private boolean pintarEscena = false;
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
	JLabel etiquetaObjetivo = new JLabel("Velocidad Objetivo");
	SpinnerNumberModel spObjetivo = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,100,0.1);
	JSpinner spinnerObjetivo = new JSpinner(spObjetivo);
	JLabel etiquetaEvitaObs = new JLabel("Evitar Obstáculos");
	SpinnerNumberModel spEvitaObs = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,0.1);
	JSpinner spinnerEvitaObs = new JSpinner(spEvitaObs);
	JLabel etiquetaVelMax = new JLabel("Velocidad Máxima");
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
		pintor.introducirBandada(getSim().getBandada());
		pintor.addMouseListener(this);
		cp.add(pintor);
		this.setJMenuBar(barraMenu);
		JPanel panelSur = new JPanel(new FlowLayout());
		JPanel panelNorte = new JPanel(new FlowLayout());
		panelSur.add(etiquetaPesoLider);
		panelSur.add(spinnerPesoLider);
		panelSur.add(etiquetaCohesion);
		panelSur.add(spinnerCohesion);
		panelSur.add(etiquetaSeparacion);
		panelSur.add(spinnerSeparacion);
		panelSur.add(etiquetaAlineacion);
		panelSur.add(spinnerAlineacion);
		panelSur.add(etiquetaObjetivo);
		panelSur.add(spinnerObjetivo);
		panelSur.add(etiquetaEvitaObs);
		panelSur.add(spinnerEvitaObs);		
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
        	if (getConfigurador().getVectorSim().size() > 0){
        		System.out.println("Existe un vector de puntos de diseño de tamaño "+getConfigurador().getVectorSim().size());
        		batch = true;
        		simulacionBatch.setEnabled(false);
//        		for(int i=0;i<getConfigurador().getVectorSim().size();i++){
////        			if (play){
//        				getSim().configurador(getConfigurador().getVectorSim().elementAt(i),
//        						getConfigurador().getNomParam());
//        				getSim().simuPorLotes();        			
////        				if (pintarEscena){
//        				this.pintor.introducirBandada(this.getSim().getBandada());
//        				this.pintor.repaint();     
//        				System.out.println("La hostia!!!!");
////        				}
////        				getSim().crearBandada(20);
//        				getSim().posicionarBandada(getSim().getPosInicial());
////        			}        				
//        				System.out.println("Ya acabó el simuporlotes");
//        				System.out.println(getSim().getNumBoidsOk());
//        		}        		
        	}        	
        }
	}
	

	public void mouseClicked(MouseEvent e) {
		if (colocandoObs){
			double pos[] = {e.getX(),e.getY()};
			double vel[] = {0,0};
			Matrix posicion = new Matrix(pos,2);
			Matrix velocidad = new Matrix(vel,2);
			Obstaculo nuevoObs = new Obstaculo(posicion,velocidad);
			this.getSim().getObstaculos().add(nuevoObs);
			pintor.introducirObstaculo(nuevoObs);
			System.out.println("Hay "+pintor.obstaculosPintar.size()+" obstáculos");
			repaint();
		}
		if (colocandoBan){
			double pos[] = {e.getX(),e.getY()};
			Matrix posicion = new Matrix(pos,2);
			this.getSim().posicionarBandada(posicion);
//			System.out.println(getSim().getBandada());
			pintor.introducirBandada(this.getSim().getBandada());
			repaint();
		}
		if (!colocandoBan && !colocandoObs){		
			Boid.setObjetivo(e.getX(),e.getY()); // cambia el coordenadas del objetivo
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
		MuestraBoids gui = new MuestraBoids();
//		Simulador simula = new Simulador();
//		gui.setSim(simula);
//		gui.getSim().setTamanoBandada(gui.getSim().getTamanoBandada());
//		gui.pintor.introducirBandada(gui.getSim().getBandada());
		Console.run(gui,1200,1000);
		int alturaPanel = gui.pintor.getHeight();
		int anchuraPanel = gui.pintor.getWidth();		
//		Bucle para crear los bordes
		int cont = 1;
		for (int k=0; k<2;k++){			
			for (int i = 0;i < alturaPanel;i=i+20){
				double posObstaculos[] = {cont,i};
				double velObstaculos[] = {0,0};				
				Matrix posiObs = new Matrix(posObstaculos,2);
				Matrix velObs = new Matrix(velObstaculos,2);
				gui.getSim().getObstaculos().add(new Obstaculo(posiObs,velObs));
			}
			cont = anchuraPanel-10;
		}
		cont = 1;
		for (int k=0; k<2;k++){			
			for (int i = 0;i < anchuraPanel;i=i+20){
				double posObstaculos[] = {i,cont};
				double velObstaculos[] = {0,0};				
				Matrix posiObs = new Matrix(posObstaculos,2);
				Matrix velObs = new Matrix(velObstaculos,2);
				gui.getSim().getObstaculos().add(new Obstaculo(posiObs,velObs));
			}
			cont = alturaPanel-15;
		}
		gui.pintor.introducirObstaculos(gui.getSim().getObstaculos());
		
		//-------------------------------------------------------------------
		//----------------BUCLE PRINCIPAL------------------------------------
		//-------------------------------------------------------------------
		
		int indMinAnt = 0;
		while (true){
			if (gui.play){
				indMinAnt = gui.getSim().moverBoids(indMinAnt);
//				gui.getSim().simuPorLotes();				
				if (gui.pintarEscena)
					gui.pintor.repaint();
			}
			if (gui.batch){
				gui.getSim().getBoidsOk().clear();
				gui.getSim().setNumBoidsOkDeseados(gui.getConfigurador().getNumBoidsOk());
				gui.getSim().setTiempoMax(gui.getConfigurador().getTMax());
				for(int i=0;i<gui.getConfigurador().getVectorSim().size();i++){
	    				gui.getSim().configurador(gui.getConfigurador().getVectorSim().elementAt(i),
	    						gui.getConfigurador().getNomParam());
	    				gui.getSim().crearBandada();
	    				gui.getSim().posicionarBandada(gui.getSim().getPosInicial());	    				
	    				gui.getSim().simuPorLotes();
	    				gui.tiempoConsumido.setText("Tardó " + gui.getSim().getTiempoInvertido() + " seg");
	    				if (gui.pintarEscena){
	    					for(int j=0;j<gui.getSim().getBoidsOk().size();j++){
	    						gui.getSim().getBoidsOk().elementAt(j).setRutaBoid(
	    								gui.getSim().mejoraRuta(
	    								gui.getSim().getBoidsOk().elementAt(j).getRutaBoid()));
	    					}	    					
	    					gui.pintor.introducirBandada(gui.getSim().getBoidsOk());
	    					gui.pintor.repaint();     	    				
	    				}	    				
//	    				gui.pintor.introducirBandada(gui.getSim().getBandada());
//	    			}        				
	    				System.out.println("Ya acabó el simuporlotes");
	    				System.out.println(gui.getSim().getNumBoidsOk());
	    		}
				gui.batch = false;
				gui.simulacionBatch.setEnabled(true);				
			}			
		}
	}
}