package repgrafica;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class Parametro extends JPanel implements ActionListener,ChangeListener{
	
	double valorIni,valorFin,valorPaso;
	
	String nombre;
	JCheckBox parametroN;
	SpinnerNumberModel spValorIni;
	JSpinner spinnerValorIni;
	SpinnerNumberModel spValorFin;
	JSpinner spinnerValorFin;
	SpinnerNumberModel spValorPaso;
	JSpinner spinnerValorPaso;	
	JLabel etiquetaInicial = new JLabel(" Valor inicial");
	JLabel etiquetaFinal = new JLabel(" Valor final");
	JLabel etiquetaPaso = new JLabel(" Paso");
	
	public Parametro(String nombre,double valIni,double valFin,double paso){
		this.nombre = nombre;
		parametroN = new JCheckBox(this.nombre);
		valorIni = valIni;
		spValorIni = new SpinnerNumberModel(valorIni,0,1000,1);
		spinnerValorIni = new JSpinner(spValorIni);
		spinnerValorIni.setEnabled(false);
		valorFin = valFin;
		spValorFin = new SpinnerNumberModel(valorFin,0,1000,1);
		spinnerValorFin = new JSpinner(spValorFin);
		spinnerValorFin.setEnabled(false);
		valorPaso = paso;
		spValorPaso = new SpinnerNumberModel(valorPaso,0,1000,1);
		spinnerValorPaso = new JSpinner(spValorPaso);
		spinnerValorPaso.setEnabled(false);
		setLayout(new GridLayout(1,7));
		add(parametroN);
		parametroN.addActionListener(this);
		add(spinnerValorIni);
		spinnerValorIni.addChangeListener(this);
		add(etiquetaInicial);
		add(spinnerValorFin);
		spinnerValorFin.addChangeListener(this);
		add(etiquetaFinal);
		add(spinnerValorPaso);
		spinnerValorPaso.addChangeListener(this);
		add(etiquetaPaso);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == parametroN){
			spinnerValorIni.setEnabled(parametroN.isSelected());
			spinnerValorFin.setEnabled(parametroN.isSelected());
			spinnerValorPaso.setEnabled(parametroN.isSelected());        	
        }		
	}
	
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == spinnerValorIni){
			valorIni = spValorIni.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerValorFin){
			valorFin = spValorFin.getNumber().doubleValue();
		}
		if (e.getSource() == spinnerValorPaso){
			valorPaso = spValorPaso.getNumber().doubleValue();
		}		
	}

	public double getValorFin() {
		return valorFin;
	}

	public void setValorFin(double valorFin) {
		this.valorFin = valorFin;
	}

	public double getValorIni() {
		return valorIni;
	}

	public void setValorIni(double valorIni) {
		this.valorIni = valorIni;
	}

	public double getValorPaso() {
		return valorPaso;
	}

	public void setValorPaso(double valorPaso) {
		this.valorPaso = valorPaso;
	}

	public boolean isSelected() {
		return parametroN.isSelected();
	}

	public String getNombre() {
		return nombre;
	}

	public void setNombre(String nombre) {
		this.nombre = nombre;
	}	
}
