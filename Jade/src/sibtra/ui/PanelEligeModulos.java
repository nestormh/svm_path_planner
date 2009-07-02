/**
 * 
 */
package sibtra.ui;

import java.awt.GridLayout;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SpringLayout;


import sibtra.util.ClasesEnPaquete;

/**
 * Panel que permitira la selección de los módulos básicos
 * @author alberto
 *
 */
public class PanelEligeModulos extends JPanel {
	
	private Class[] arrClasMotor;
	private JComboBox jcombMotor;
	private Class[] arrClasDir;
	private Class[] arrClasVel;
	private Class[] arrClasDectObs;
	private JComboBox jcombDireccion;
	private JComboBox jcombVelocidad;
	private JComboBox jcombDetecObstaculos;

	public PanelEligeModulos() {
		super(new GridLayout(0,2,10,10));
		
		arrClasMotor=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.Motor", "sibtra.ui.modulos");
		String[] arrNomClasMotor=nombreClases(arrClasMotor);
			
		arrClasDir=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.CalculoDireccion", "sibtra.ui.modulos");
		String[] arrNomClasDir=nombreClases(arrClasDir);
			
		arrClasVel=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.CalculoVelocidad", "sibtra.ui.modulos");
		String[] arrNomClasVel=nombreClases(arrClasVel);
			
		arrClasDectObs=ClasesEnPaquete.clasesImplementan("sibtra.ui.modulos.DetectaObstaculos", "sibtra.ui.modulos");
		String[] arrNomClasDectObs=nombreClases(arrClasDectObs);
			
		
		//ponemos etiqueta y selector
		add(new JLabel("Motor Elegido",JLabel.TRAILING));
		jcombMotor=new JComboBox(arrNomClasMotor);
		add(jcombMotor);
		
		add(new JLabel("Calculo Direccion",JLabel.TRAILING));
		jcombDireccion=new JComboBox(arrNomClasDir);
		add(jcombDireccion);
		
		add(new JLabel("Calculo Velocidad",JLabel.TRAILING));
		jcombVelocidad=new JComboBox(arrNomClasVel);
		add(jcombVelocidad);
		
		add(new JLabel("Motor Elegido",JLabel.TRAILING));
		jcombDetecObstaculos=new JComboBox(arrNomClasDectObs);
		add(jcombDetecObstaculos);
		
		
		
	}
	
	private String[] nombreClases(Class[] arrClas) {
		String[] resp=new String[arrClasMotor.length];
		for(int i=0; i<arrClasMotor.length; i++ ) {
			try {
				Method mn=arrClas[i].getMethod("getNombre", (Class[])null);
				Object ob=arrClas[i].newInstance();
				resp[i]=(String)mn.invoke(ob, (Object[])null);
			} catch (Exception e) {
				System.err.println("Error al invocar getNombre() de clase "+arrClas[i].getName());
				e.printStackTrace();
			}
		}
		return resp;
	}
}
