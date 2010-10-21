package repgrafica;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerListModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import boids.*;

import com.bruceeckel.swing.Console;

//import com.sun.xml.internal.ws.api.server.Container;

public class ConfiguraParam extends JApplet implements ChangeListener,ActionListener{	
	Vector <DesignPoint> vectorSim= new Vector<DesignPoint>();
	// Etiquetas para todos los parámetros en general
	JLabel valorInicial = new JLabel("valor inicial");
	JLabel valorFinal = new JLabel("valor final");
	JLabel paso = new JLabel("Paso");
	JTextArea numIteraciones = new JTextArea();
	double radioObstaculoIni,radioObstaculoFin,radioObstaculoPaso;
	double radioCohesionIni,radioCohesionFin,radioCohesionPaso;
	double radioSeparacionIni,radioSeparacionFin,radioSeparacionPaso;
	double radioAlineacionIni,radioAlineacionFin,radioAlineacionPaso;
	double pesoObjetivoIni,pesoObjetivoFin,pesoObjetivoPaso;
	double pesoObstaculoIni,pesoObstaculoFin,pesoObstaculoPaso;
	double pesoLiderIni,pesoLiderFin,pesoLiderPaso;
	double velMaxIni,velMaxFin,velMaxPaso;
	int numBoidsIni,numBoidsFin,numBoidsPaso;
	// Def para radio Obstáculo
	JCheckBox radioObstaculo = new JCheckBox("Radio Obstáculos");
	SpinnerNumberModel spRadObsIni = new SpinnerNumberModel(Boid.getRadioObstaculo(),0,1000,1);
	JSpinner spinnerRadObsIni = new JSpinner(spRadObsIni);
	SpinnerNumberModel spRadObsFin = new SpinnerNumberModel(Boid.getRadioObstaculo(),0,1000,1);
	JSpinner spinnerRadObsFin = new JSpinner(spRadObsFin);
	SpinnerNumberModel spRadObsPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerRadObsP = new JSpinner(spRadObsPaso);	
	// Def para radio Cohesión	
	JCheckBox radioCohesion = new JCheckBox("Radio Cohesión");	
	SpinnerNumberModel spRadCoheIni = new SpinnerNumberModel(Boid.getRadioCohesion(),0,1000,1);
	JSpinner spinnerRadCoheIni = new JSpinner(spRadCoheIni);
	SpinnerNumberModel spRadCoheFin = new SpinnerNumberModel(Boid.getRadioCohesion(),0,1000,1);
	JSpinner spinnerRadCoheFin = new JSpinner(spRadCoheFin);
	SpinnerNumberModel spRadCohePaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerRadCohePaso = new JSpinner(spRadCohePaso);
	// Def para radio de Separación
	JCheckBox radioSeparacion = new JCheckBox("Radio Separación");	
	SpinnerNumberModel spRadSepIni = new SpinnerNumberModel(Boid.getRadioSeparacion(),0,1000,1);
	JSpinner spinnerRadSepIni = new JSpinner(spRadSepIni);
	SpinnerNumberModel spRadSepFin = new SpinnerNumberModel(Boid.getRadioSeparacion(),0,1000,1);
	JSpinner spinnerRadSepFin = new JSpinner(spRadSepFin);
	SpinnerNumberModel spRadSepPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerRadSepPaso = new JSpinner(spRadSepPaso);
	// Def para radio de Alineación
	JCheckBox radioAlineacion = new JCheckBox("Radio Alineación");
	SpinnerNumberModel spRadAliIni = new SpinnerNumberModel(Boid.getRadioAlineacion(),0,1000,1);
	JSpinner spinnerRadAliIni = new JSpinner(spRadAliIni);
	SpinnerNumberModel spRadAliFin = new SpinnerNumberModel(Boid.getRadioAlineacion(),0,1000,1);
	JSpinner spinnerRadAliFin = new JSpinner(spRadAliFin);
	SpinnerNumberModel spRadAliPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerRadAliPaso = new JSpinner(spRadAliPaso);
	// Def para peso de Cohesión
	JCheckBox pesoCohesion = new JCheckBox("Peso Cohesión");
	SpinnerNumberModel spPesoCoheIni = new SpinnerNumberModel(Boid.getPesoCohesion(),0,1000,1);
	JSpinner spinnerPesoCoheIni = new JSpinner(spPesoCoheIni);
	SpinnerNumberModel spPesoCoheFin = new SpinnerNumberModel(Boid.getPesoCohesion(),0,1000,1);
	JSpinner spinnerPesoCoheFin = new JSpinner(spPesoCoheFin);
	SpinnerNumberModel spPesoCohePaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerPesoCohePaso = new JSpinner(spPesoCohePaso);
	// Def para peso de Separación
	JCheckBox pesoSeparacion = new JCheckBox("Peso Separación");
	SpinnerNumberModel spPesoSepIni = new SpinnerNumberModel(Boid.getPesoSeparacion(),0,1000,1);
	JSpinner spinnerPesoSepIni = new JSpinner(spPesoSepIni);
	SpinnerNumberModel spPesoSepFin = new SpinnerNumberModel(Boid.getPesoSeparacion(),0,1000,1);
	JSpinner spinnerPesoSepFin = new JSpinner(spPesoSepFin);
	SpinnerNumberModel spPesoSepPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerPesoSepPaso = new JSpinner(spPesoSepPaso);
	// Def para peso de Alineación
	JCheckBox pesoAlineacion = new JCheckBox("Peso Alineación");
	SpinnerNumberModel spPesoAliIni = new SpinnerNumberModel(Boid.getPesoAlineacion(),0,1000,1);
	JSpinner spinnerPesoAliIni = new JSpinner(spPesoAliIni);
	SpinnerNumberModel spPesoAliFin = new SpinnerNumberModel(Boid.getPesoAlineacion(),0,1000,1);
	JSpinner spinnerPesoAliFin = new JSpinner(spPesoAliFin);
	SpinnerNumberModel spPesoAliPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerPesoAliPaso = new JSpinner(spPesoAliPaso);
	// Def para peso de Objetivo
	JCheckBox pesoObjetivo = new JCheckBox("Peso Objetivo");
	SpinnerNumberModel spPesoObjIni = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,1000,1);
	JSpinner spinnerPesoObjIni = new JSpinner(spPesoObjIni);
	SpinnerNumberModel spPesoObjFin = new SpinnerNumberModel(Boid.getPesoObjetivo(),0,1000,1);
	JSpinner spinnerPesoObjFin = new JSpinner(spPesoObjFin);
	SpinnerNumberModel spPesoObjPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerPesoObjPaso = new JSpinner(spPesoObjPaso);
	// Def para peso de Obstáculo
	JCheckBox pesoObstaculo = new JCheckBox("Peso Obstáculos");
	SpinnerNumberModel spPesoObstIni = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,1);
	JSpinner spinnerPesoObstIni = new JSpinner(spPesoObstIni);
	SpinnerNumberModel spPesoObstFin = new SpinnerNumberModel(Boid.getPesoObstaculo(),0,1000,1);
	JSpinner spinnerPesoObstFin = new JSpinner(spPesoObstFin);
	SpinnerNumberModel spPesoObstPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerPesoObstPaso = new JSpinner(spPesoObstPaso);
	// Def para el peso del lider
	JCheckBox pesoLider = new JCheckBox("Peso Lider");
	SpinnerNumberModel spPesoLiderIni = new SpinnerNumberModel(Boid.getPesoLider(),0,1000,1);
	JSpinner spinnerPesoLiderIni = new JSpinner(spPesoLiderIni);
	SpinnerNumberModel spPesoLiderFin = new SpinnerNumberModel(Boid.getPesoLider(),0,1000,1);
	JSpinner spinnerPesoLiderFin = new JSpinner(spPesoLiderFin);
	SpinnerNumberModel spPesoLiderPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerPesoLiderPaso = new JSpinner(spPesoLiderPaso);
	// Def para la velocidad máxima
	JCheckBox velMax = new JCheckBox("Velocidad Máxima");
	SpinnerNumberModel spVelMaxIni = new SpinnerNumberModel(Boid.getVelMax(),0,1000,1);
	JSpinner spinnerVelMaxIni = new JSpinner(spVelMaxIni);
	SpinnerNumberModel spVelMaxFin = new SpinnerNumberModel(Boid.getVelMax(),0,1000,1);
	JSpinner spinnerVelMaxFin = new JSpinner(spVelMaxFin);
	SpinnerNumberModel spVelMaxPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerVelMaxPaso = new JSpinner(spVelMaxPaso);
//	 Def para el número de boids
	JCheckBox numBoids = new JCheckBox("Número de boids");
	SpinnerNumberModel spNumBoidsIni = new SpinnerNumberModel(20,0,1000,1);
	JSpinner spinnerNumBoidsIni = new JSpinner(spNumBoidsIni);
	SpinnerNumberModel spNumBoidsFin = new SpinnerNumberModel(20,0,1000,1);
	JSpinner spinnerNumBoidsFin = new JSpinner(spNumBoidsFin);
	SpinnerNumberModel spNumBoidsPaso = new SpinnerNumberModel(1,0,1000,1);
	JSpinner spinnerNumBoidsPaso = new JSpinner(spNumBoidsPaso);
	
	// Parámetros para finalizar la simulación
	
	SpinnerNumberModel spNumBoidsOk = new SpinnerNumberModel(0,0,1000,1);
	JSpinner spinnerNumBoidsOk = new JSpinner(spNumBoidsOk);
	JLabel numBoidsOk = new JLabel("Número de boids con éxito");
	SpinnerNumberModel spTmax = new SpinnerNumberModel(0,0,1000,1);
	JSpinner spinnerTmax = new JSpinner(spTmax);
	JLabel tMax = new JLabel("Tiempo máximo para la simulación");
	JButton configurar = new JButton("Configurar");
	
	public ConfiguraParam(){
		vectorSim.add(new DesignPoint());
		radioObstaculoIni = spRadObsIni.getNumber().doubleValue();
		radioObstaculoFin = spRadObsFin.getNumber().doubleValue();
		radioObstaculoPaso = spRadObsPaso.getNumber().doubleValue();
		radioCohesionIni = spRadCoheIni.getNumber().doubleValue();
		radioCohesionFin = spRadCoheFin.getNumber().doubleValue();
		radioCohesionPaso = spRadCohePaso.getNumber().doubleValue();
		radioSeparacionIni = spRadSepIni.getNumber().doubleValue();
		radioSeparacionFin = spRadSepFin.getNumber().doubleValue();
		radioSeparacionPaso = spRadSepPaso.getNumber().doubleValue();
		radioAlineacionIni = spRadAliIni.getNumber().doubleValue();
		radioAlineacionFin = spRadAliFin.getNumber().doubleValue();
		radioAlineacionPaso = spRadAliPaso.getNumber().doubleValue();
		pesoObjetivoIni = spPesoObjIni.getNumber().doubleValue();
		pesoObjetivoFin = spPesoObjFin.getNumber().doubleValue();
		pesoObjetivoPaso = spPesoObjPaso.getNumber().doubleValue();
		pesoObstaculoIni = spPesoObstIni.getNumber().doubleValue();
		pesoObstaculoFin = spPesoObstFin.getNumber().doubleValue();
		pesoObstaculoPaso = spPesoObstPaso.getNumber().doubleValue();
		pesoLiderIni = spPesoLiderIni.getNumber().doubleValue();
		pesoLiderFin = spPesoLiderFin.getNumber().doubleValue();
		pesoLiderPaso = spPesoLiderPaso.getNumber().doubleValue();
		velMaxIni = spVelMaxIni.getNumber().doubleValue();
		velMaxFin = spVelMaxFin.getNumber().doubleValue();
		velMaxPaso = spVelMaxPaso.getNumber().doubleValue();
		numBoidsIni = spNumBoidsIni.getNumber().intValue();
		numBoidsFin = spNumBoidsFin.getNumber().intValue();
		numBoidsPaso = spNumBoidsPaso.getNumber().intValue();
	}
	
	public void init(){
		Container cp = getContentPane();
		JPanel panel = new JPanel(new GridLayout(9,7));
		JPanel panelSur = new JPanel(new FlowLayout());
		//Radio para los obstáculos
		panel.add(radioObstaculo);
		radioObstaculo.addActionListener(this);
		panel.add(spinnerRadObsIni);
		spinnerRadObsIni.setEnabled(false);
		spinnerRadObsIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerRadObsFin);
		spinnerRadObsFin.setEnabled(false);
		spinnerRadObsFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerRadObsP);
		spinnerRadObsP.setEnabled(false);
		spinnerRadObsP.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Radio para la cohesión
		panel.add(radioCohesion);
		radioCohesion.addActionListener(this);
		panel.add(spinnerRadCoheIni);
		spinnerRadCoheIni.setEnabled(false);
		spinnerRadCoheIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerRadCoheFin);
		spinnerRadCoheFin.setEnabled(false);
		spinnerRadCoheFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerRadCohePaso);
		spinnerRadCohePaso.setEnabled(false);
		spinnerRadCohePaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Radio para la separación
		panel.add(radioSeparacion);
		radioSeparacion.addActionListener(this);
		panel.add(spinnerRadSepIni);
		spinnerRadSepIni.setEnabled(false);
		spinnerRadSepIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerRadSepFin);
		spinnerRadSepFin.setEnabled(false);
		spinnerRadSepFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerRadSepPaso);
		spinnerRadSepPaso.setEnabled(false);
		spinnerRadSepPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Radio para la alineación
		panel.add(radioAlineacion);
		radioAlineacion.addActionListener(this);
		panel.add(spinnerRadAliIni);
		spinnerRadAliIni.setEnabled(false);
		spinnerRadAliIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerRadAliFin);
		spinnerRadAliFin.setEnabled(false);
		spinnerRadAliFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerRadAliPaso);
		spinnerRadAliPaso.setEnabled(false);
		spinnerRadAliPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Peso para el objetivo
		panel.add(pesoObjetivo);
		pesoObjetivo.addActionListener(this);
		panel.add(spinnerPesoObjIni);
		spinnerPesoObjIni.setEnabled(false);
		spinnerPesoObjIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerPesoObjFin);
		spinnerPesoObjFin.setEnabled(false);
		spinnerPesoObjFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerPesoObjPaso);
		spinnerPesoObjPaso.setEnabled(false);
		spinnerPesoObjPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Peso para los obstáculos
		panel.add(pesoObstaculo);
		pesoObstaculo.addActionListener(this);
		panel.add(spinnerPesoObstIni);
		spinnerPesoObstIni.setEnabled(false);
		spinnerPesoObstIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerPesoObstFin);
		spinnerPesoObstFin.setEnabled(false);
		spinnerPesoObstFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerPesoObstPaso);
		spinnerPesoObstPaso.setEnabled(false);
		spinnerPesoObstPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Peso para el liderazgo
		panel.add(pesoLider);
		pesoLider.addActionListener(this);
		panel.add(spinnerPesoLiderIni);
		spinnerPesoLiderIni.setEnabled(false);
		spinnerPesoLiderIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerPesoLiderFin);
		spinnerPesoLiderFin.setEnabled(false);
		spinnerPesoLiderFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerPesoLiderPaso);
		spinnerPesoLiderPaso.setEnabled(false);
		spinnerPesoLiderPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// velocidad máxima
		panel.add(velMax);
		velMax.addActionListener(this);
		panel.add(spinnerVelMaxIni);
		spinnerVelMaxIni.setEnabled(false);
		spinnerVelMaxIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerVelMaxFin);
		spinnerVelMaxFin.setEnabled(false);
		spinnerVelMaxFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerVelMaxPaso);
		spinnerVelMaxPaso.setEnabled(false);
		spinnerVelMaxPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
//		 Número de Boids
		panel.add(numBoids);
		numBoids.addActionListener(this);
		panel.add(spinnerNumBoidsIni);
		spinnerNumBoidsIni.setEnabled(false);
		spinnerNumBoidsIni.addChangeListener(this);
		panel.add(new JLabel(" Valor inicial"));
		panel.add(spinnerNumBoidsFin);
		spinnerNumBoidsFin.setEnabled(false);
		spinnerNumBoidsFin.addChangeListener(this);
		panel.add(new JLabel(" Valor final"));
		panel.add(spinnerNumBoidsPaso);
		spinnerNumBoidsPaso.setEnabled(false);
		spinnerNumBoidsPaso.addChangeListener(this);
		panel.add(new JLabel("Paso"));
		// Parámetros de parada de la simulación
		panelSur.add(spinnerNumBoidsOk);
		spinnerNumBoidsOk.addChangeListener(this);
		panelSur.add(numBoidsOk);
		panelSur.add(spinnerTmax);
		spinnerTmax.addChangeListener(this);
		panelSur.add(tMax);
		panelSur.add(configurar);
		configurar.addActionListener(this);
		cp.add(panel);
		cp.add(BorderLayout.SOUTH,panelSur);
		
	}
	
	public void stateChanged(ChangeEvent e) {
		
		//----------radio obstáculos------------
		if (e.getSource() == spinnerRadObsIni){
			radioObstaculoIni = spRadObsIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadObsFin){
			radioObstaculoFin = spRadObsFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadObsP){
			radioObstaculoPaso = spRadObsPaso.getNumber().doubleValue();
		}
		//----------radio cohesión---------------		
		if (e.getSource() == spinnerRadCoheIni){
			radioCohesionIni = spRadCoheIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadCoheFin){
			radioCohesionFin = spRadCoheFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadCohePaso){
			radioCohesionPaso = spRadCohePaso.getNumber().doubleValue();
		}
//		----------radio separación---------------		
		if (e.getSource() == spinnerRadSepIni){
			radioSeparacionIni = spRadSepIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadSepFin){
			radioSeparacionFin = spRadSepFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadSepPaso){
			radioSeparacionPaso = spRadSepPaso.getNumber().doubleValue();
		}
//		----------radio alineación---------------		
		if (e.getSource() == spinnerRadAliIni){
			radioAlineacionIni = spRadAliIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadAliFin){
			radioAlineacionFin = spRadAliFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerRadAliPaso){
			radioAlineacionPaso = spRadAliPaso.getNumber().doubleValue();
		}
//		----------peso objetivo---------------		
		if (e.getSource() == spinnerPesoObjIni){
			pesoObjetivoIni = spPesoObjIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerPesoObjFin){
			pesoObjetivoFin = spPesoObjFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerPesoObjPaso){
			pesoObjetivoPaso = spPesoObjPaso.getNumber().doubleValue();
		}
//		----------peso obstáculos---------------		
		if (e.getSource() == spinnerPesoObstIni){
			pesoObstaculoIni = spPesoObstIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerPesoObstFin){
			pesoObstaculoFin = spPesoObstFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerPesoObstPaso){
			pesoObstaculoPaso = spPesoObstPaso.getNumber().doubleValue();
		}
//		----------peso lider---------------		
		if (e.getSource() == spinnerPesoLiderIni){
			pesoLiderIni = spPesoLiderIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerPesoLiderFin){
			pesoLiderFin = spPesoLiderFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerPesoLiderPaso){
			pesoLiderPaso = spPesoLiderPaso.getNumber().doubleValue();
		}
//		----------velocidad máxima---------------		
		if (e.getSource() == spinnerVelMaxIni){
			velMaxIni = spVelMaxIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerVelMaxFin){
			velMaxFin = spVelMaxFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerVelMaxPaso){
			velMaxPaso = spVelMaxPaso.getNumber().doubleValue();
		}
//		----------número de boids---------------		
		if (e.getSource() == spinnerNumBoidsIni){
			numBoidsIni = spNumBoidsIni.getNumber().intValue();
		}
		if (e.getSource() == spinnerNumBoidsFin){
			numBoidsFin = spNumBoidsFin.getNumber().intValue();
		}
		if (e.getSource() == spinnerNumBoidsPaso){
			numBoidsPaso = spNumBoidsPaso.getNumber().intValue();
		}
	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == radioObstaculo){
			spinnerRadObsIni.setEnabled(radioObstaculo.isSelected());
			spinnerRadObsFin.setEnabled(radioObstaculo.isSelected());
			spinnerRadObsP.setEnabled(radioObstaculo.isSelected());        	
        }
		if (e.getSource() == radioCohesion){
			spinnerRadCoheIni.setEnabled(radioCohesion.isSelected());
			spinnerRadCoheFin.setEnabled(radioCohesion.isSelected());
			spinnerRadCohePaso.setEnabled(radioCohesion.isSelected());        	
        }
		if (e.getSource() == radioSeparacion){
			spinnerRadSepIni.setEnabled(radioSeparacion.isSelected());
			spinnerRadSepFin.setEnabled(radioSeparacion.isSelected());
			spinnerRadSepPaso.setEnabled(radioSeparacion.isSelected());        	
        }
		if (e.getSource() == radioAlineacion){
			spinnerRadAliIni.setEnabled(radioAlineacion.isSelected());
			spinnerRadAliFin.setEnabled(radioAlineacion.isSelected());
			spinnerRadAliPaso.setEnabled(radioAlineacion.isSelected());        	
        }
		if (e.getSource() == pesoObjetivo){
			spinnerPesoObjIni.setEnabled(pesoObjetivo.isSelected());
			spinnerPesoObjFin.setEnabled(pesoObjetivo.isSelected());
			spinnerPesoObjPaso.setEnabled(pesoObjetivo.isSelected());        	
        }
		if (e.getSource() == pesoObstaculo){
			spinnerPesoObstIni.setEnabled(pesoObstaculo.isSelected());
			spinnerPesoObstFin.setEnabled(pesoObstaculo.isSelected());
			spinnerPesoObstPaso.setEnabled(pesoObstaculo.isSelected());        	
        }
		if (e.getSource() == pesoLider){
			spinnerPesoLiderIni.setEnabled(pesoLider.isSelected());
			spinnerPesoLiderFin.setEnabled(pesoLider.isSelected());
			spinnerPesoLiderPaso.setEnabled(pesoLider.isSelected());        	
        }
		if (e.getSource() == velMax){
			spinnerVelMaxIni.setEnabled(velMax.isSelected());
			spinnerVelMaxFin.setEnabled(velMax.isSelected());
			spinnerVelMaxPaso.setEnabled(velMax.isSelected());        	
        }
		if (e.getSource() == numBoids){
			spinnerNumBoidsIni.setEnabled(numBoids.isSelected());
			spinnerNumBoidsFin.setEnabled(numBoids.isSelected());
			spinnerNumBoidsPaso.setEnabled(numBoids.isSelected());        	
        }
		
		//---------Configuración del vector de puntos de diseño---------------
		
		if (e.getSource() == configurar){
			// Cuando se hace click en el botón configurar se genera el vector de 
			// puntos de diseño
			vectorSim.clear();
			//----------radio obstáculo------------------
			
			if (radioObstaculo.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = radioObstaculoIni;
					while (valorParam < radioObstaculoFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setRadioObstaculo(valorParam);
						vecAux.add(nuevoPunto);	
						valorParam = valorParam + radioObstaculoPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = radioObstaculoIni;
						while (valorParam < radioObstaculoFin){
							vectorSim.elementAt(i).setRadioObstaculo(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + radioObstaculoPaso;
						}						
					}
				}				
				vectorSim = vecAux;
			}
			
//			----------radio cohesión------------------
			
			if (radioCohesion.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = radioCohesionIni;
					while (valorParam < radioCohesionFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setRadioCohesion(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + radioCohesionPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = radioCohesionIni;
						while (valorParam < radioCohesionFin){
							vectorSim.elementAt(i).setRadioCohesion(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + radioCohesionPaso;
						}						
					}
				}
				vectorSim = vecAux;
			}
			
			//---------radio separación-------------------
			
			if (radioSeparacion.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = radioSeparacionIni;
					while (valorParam < radioSeparacionFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setRadioSeparacion(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + radioSeparacionPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = radioSeparacionIni;
						while (valorParam < radioSeparacionFin){
							vectorSim.elementAt(i).setRadioSeparacion(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + radioSeparacionPaso;
						}						
					}
				}
				
				vectorSim = vecAux;
			}
			
			//-----------Radio alineación---------------
			
			if (radioAlineacion.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = radioAlineacionIni;
					while (valorParam < radioAlineacionFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setRadioAlineacion(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + radioAlineacionPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = radioAlineacionIni;
						while (valorParam < radioAlineacionFin){
							vectorSim.elementAt(i).setRadioAlineacion(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + radioAlineacionPaso;
						}						
					}
				}
				
				vectorSim = vecAux;
			}
			
			// ----------Peso objetivo-----------------
			
			if (pesoObjetivo.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = pesoObjetivoIni;
					while (valorParam < pesoObjetivoFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setPesoObjetivo(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + pesoObjetivoPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = pesoObjetivoIni;
						while (valorParam < pesoObjetivoFin){
							vectorSim.elementAt(i).setPesoObjetivo(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + pesoObjetivoPaso;
						}						
					}
				}				
				vectorSim = vecAux;
			}
			
			//---------Peso obstáculo---------------
			
			if (pesoObstaculo.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = pesoObstaculoIni;
					while (valorParam < pesoObstaculoFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setPesoObstaculo(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + pesoObstaculoPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = pesoObstaculoIni;
						while (valorParam < pesoObstaculoFin){
							vectorSim.elementAt(i).setPesoObstaculo(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + pesoObstaculoPaso;
						}						
					}
				}				
				vectorSim = vecAux;
			}
			
			//------------Peso lider--------------------
			
			if (pesoLider.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = pesoLiderIni;
					while (valorParam < pesoLiderFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setPesoLider(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + pesoLiderPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = pesoLiderIni;
						while (valorParam < pesoLiderFin){
							vectorSim.elementAt(i).setPesoLider(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + pesoLiderPaso;
						}						
					}
				}
				vectorSim = vecAux;
			}
			
			// --------Velocidad máxima---------------------
			
			if (velMax.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					double valorParam = velMaxIni;
					while (valorParam < velMaxFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setVelMax(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + velMaxPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						double valorParam = velMaxIni;
						while (valorParam < velMaxFin){
							vectorSim.elementAt(i).setVelMax(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + velMaxPaso;
						}						
					}
				}				
				vectorSim = vecAux;
			}
			
			// ---------número de boids----------------------
			
			if (numBoids.isSelected()){
				Vector <DesignPoint> vecAux =new Vector <DesignPoint>();
				if (vectorSim.size() <= 0){
					int valorParam = numBoidsIni;
					while (valorParam < numBoidsFin){
						DesignPoint nuevoPunto = new DesignPoint();
						nuevoPunto.setNumBoids(valorParam);
						vecAux.add(nuevoPunto);						
						valorParam = valorParam + numBoidsPaso;
					}
				}
				else{
					for (int i=0;i<vectorSim.size();i++){
						int valorParam = numBoidsIni;
						while (valorParam < numBoidsFin){
							vectorSim.elementAt(i).setNumBoids(valorParam);
							vecAux.add(vectorSim.elementAt(i));						
							valorParam = valorParam + numBoidsPaso;
						}						
					}
				}				
				vectorSim = vecAux;
			}
			System.out.println(vectorSim.size());
			
		}
	}
	
	public static void main(String[] args){
		ConfiguraParam configurador = new ConfiguraParam();
		Console.run(configurador,1000,300);
	}

}
